import sys
sys.dont_write_bytecode = True
import math
import time
import argparse
from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import tempfile

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe

from models import Encoder, Decoder


if torch.cuda.device_count() < 2:
    print('Need at least two GPU devices for this tutorial')
    sys.exit(0)
    
# Load and batch data 
def data_process(vocab, tokenizer, raw_text_iter):
  data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: Tensor, bsz: int, device: int):
    # Divide the dataset into ``bsz`` parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the ``bsz` batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.cuda(device)

def get_data(batch_size: int = 20, eval_batch_size: int = 10, device: int = 0):
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(vocab, tokenizer, train_iter)
    val_data = data_process(vocab, tokenizer, val_iter)
    test_data = data_process(vocab, tokenizer, test_iter)

    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)
    
    return train_data, val_data, test_data, vocab


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def main():

    # Model scale and pipe initialization
    bptt = args.bptt
    emsize = args.emsize
    nhid = args.nhid
    nlayers = args.nlayers
    nhead = args.nhead
    dropout = args.dropout
    lr = args.lr
    epochs = args.epochs
    chunks = args.chunks
    do_train = args.do_train
    do_eval = args.do_eval
    
    train_data, val_data, test_data, vocab = get_data()
    ntokens = len(vocab) # the size of vocabulary
    
    def get_batch(source, i):
        # Source (N X T) where T is the chunk size
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # (B X T) where B is the batch size
        target = source[i+1:i+1+seq_len].view(-1) # (B*T) the corresponding next sentences
        # Need batch dimension first for pipeline parallelism.
        return data.t(), target

    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            # Specifying _transports and _channels is a workaround and we no longer
            # will have to specify _transports and _channels for PyTorch
            # versions >= 1.8.1
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )

    num_gpus = torch.cuda.device_count()
    partition_len = ((nlayers - 1) // num_gpus) + 1
    print("partition (block) size: {}".format(partition_len))
    start_gpu = 0

    # Add encoder in the beginning.
    tmp_list = [Encoder(ntokens, emsize, dropout).cuda(start_gpu)]
    module_list = []

    # Add all the necessary transformer blocks.
    for i in range(nlayers):
        transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        if i != 0 and i % (partition_len) == 0:
            module_list.append(nn.Sequential(*tmp_list))
            tmp_list = []
        device = i // (partition_len) + start_gpu
        print("partition {}-th layer on device {}".format(i, device))
        tmp_list.append(transformer_block.to(device))

    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize).cuda(num_gpus - 1 + start_gpu))
    module_list.append(nn.Sequential(*tmp_list))

    # Build the pipeline.
    model = Pipe(torch.nn.Sequential(*module_list), chunks=chunks)
    print ('Total parameters in model: {:,}'.format(get_total_params(model)))

    # Run the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(vocab)

        # Train only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, train_data.size(0) - 1)

        for batch, i in tqdm(enumerate(range(0, nbatches, bptt)), total=nbatches // bptt):
            data, targets = get_batch(train_data, i) # (B X T), (B*T)
            # print('input text[0]: ', vocab.lookup_tokens(data[0].tolist()))
            # print('target text[0]: ', vocab.lookup_tokens(targets[:bptt].tolist()))
            optimizer.zero_grad()
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            output = model(data).local_value() # (B X T X C) where C is vocab size
            # print('output text[0]: ', vocab.lookup_tokens(output[0].argmax(dim=-1).detach().tolist()))
            # Need to move targets to the device where the output of the
            # pipeline resides.
            loss = criterion(output.view(-1, ntokens), targets.to(output.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 10
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          epoch, batch, nbatches // bptt, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss),
                      ))
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(vocab)
        # Evaluate only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, data_source.size(0) - 1)
        with torch.no_grad():
            for i in tqdm(range(0, nbatches, bptt)):
                data, targets = get_batch(data_source, i)
                output = eval_model(data).local_value()
                output_flat = output.view(-1, ntokens)
                # Need to move targets to the device where the output of the
                # pipeline resides.
                total_loss += len(data) * criterion(output_flat, targets.to(output.device)).item()
        return total_loss / (len(data_source) - 1)

    best_model = None
    if do_train:
        best_val_loss = float("inf")
        print("***** Running training *****")
        print(f"  Num examples = {len(train_data)}")
        print(f"  Num Epochs = {epochs}")
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(model, val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

            scheduler.step()

    best_model = model if best_model is None else best_model
        
    if do_eval:
        print("***** Running evaluation *****")
        print(f"  Num examples = {len(test_data)}")
        
        test_loss = evaluate(best_model, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--bptt', type=int, default=25, help='batch size')
    parser.add_argument('--emsize', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=4096, help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=12, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=16, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--lr', type=float, default=5.0, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--chunks', type=int, default=8, help='number of micro-batches')
    parser.add_argument('--profile', action='store_true', help='profile training')
    parser.add_argument('--do_train', action='store_true', help='do training')
    parser.add_argument('--do_eval', action='store_true', help='do evaluation')

    args = parser.parse_args()
    
    if args.profile:
        with torch.cuda.profiler.profile():
            main()
    else:
        main()
