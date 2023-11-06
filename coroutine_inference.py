import sys
sys.dont_write_bytecode = True
import math
import time
import argparse
from tqdm import tqdm

import asyncio
from torch.cuda.amp import autocast
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import tempfile

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from torch.distributed import rpc

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


class PipelineStage(nn.Module):
    def __init__(self, layers, device):
        super(PipelineStage, self).__init__()
        self.layers = nn.Sequential(*layers).to(device)
        self.device = device

    def forward(self, x):
        return self.layers(x)


def main():
    # Model scale and pipe initialization
    bptt = args.bptt
    emsize = args.emsize
    nhid = args.nhid
    nlayers = args.nlayers
    nhead = args.nhead
    dropout = args.dropout
    
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

    # Add encoder in the beginning.
    tmp_list = [Encoder(ntokens, emsize, dropout)]
    stages = []

    # Add all the necessary transformer blocks.
    for i in range(nlayers):
        transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        if i != 0 and i % (partition_len) == 0:
            # Create a new pipeline stage
            stage_device = i // (partition_len) - 1
            print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device))
            stages.append(PipelineStage(tmp_list, stage_device))
            tmp_list = []
            
        tmp_list.append(transformer_block)

    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize))
    stages.append(PipelineStage(tmp_list, num_gpus - 1))
    print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], num_gpus - 1))

    # Run the model
    criterion = nn.CrossEntropyLoss()
    
    async def evaluate_stage(stage, queue_in, queue_out, device, next_device=None):
        while True:
            data, targets = await queue_in.get()
            if data is None:  # None is sent as a signal to shut down.
                await queue_out.put((None, None))
                break
            with torch.no_grad():
                data = data.to(device)  # Move data to the current stage's device.
                output = stage(data)
                if next_device:
                    output = output.to(next_device)  # Move output to the next stage's device.
            await queue_out.put((output, targets))
            
    async def producer(queue, data_source, bptt):
        for i in tqdm(range(0, data_source.size(0) - 1, bptt)):
            data, targets = get_batch(data_source, i)
            await queue.put((data, targets))
        await queue.put((None, None))  # Signal the end of the dataset
        
    async def consumer(queue, ntokens):
        total_loss = 0.
        nitems = 0
        while True:
            output, targets = await queue.get()
            if output is None:
                break
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets).item()
            nitems += output_flat.size(0)
        return total_loss / nitems
    
    async def evaluate(stages, data_source):
        queues = [asyncio.Queue() for _ in range(len(stages) + 1)]
        
        # Start the producer coroutine
        producer_coro = producer(queues[0], data_source, bptt)

        # Start stage coroutines
        stages_coros = [evaluate_stage(
            stages[i], 
            queues[i], 
            queues[i + 1], 
            device=i, 
            next_device=i+1 if i < len(stages) - 1 else None,
        ) for i in range(len(stages))]

        # Start the consumer coroutine
        consumer_coro = consumer(queues[-1], len(vocab))

        # Run the coroutines
        coros = [producer_coro] + stages_coros + [consumer_coro]
        # completed, pending = await asyncio.wait(coros, return_when=asyncio.FIRST_EXCEPTION)
        tasks = [asyncio.create_task(coro) for coro in coros]
        completed, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        # Handle exceptions if any
        for task in completed:
            if task.exception():
                print("Exception:", task.exception())
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                # Rethrow the exception
                raise task.exception()
        
        # Get the result from the consumer
        loss = await consumer_coro
        print(f"Test Loss: {loss}")
        # return loss

            
    # # Assuming that model is a list of partitions, each on its respective GPU.
    # async def evaluate_partition(data, model_partition, next_partition=None):
    #     model_partition.eval()
    #     # Move data to the GPU where this partition of the model is.
    #     data_gpu = data.to(f'cuda:{model_partition.device}')
    #     # Forward pass through this partition.
    #     with torch.no_grad():
    #         output = model_partition(data_gpu)
    #     # Pass output to the next partition if there is one.
    #     if next_partition is not None:
    #         return await evaluate_partition(output, next_partition)
    #     else:
    #         # On the last partition, return the output to be collected.
    #         return output.cpu()

    # async def evaluate_coroutine(model, data_source, ntokens):
    #     total_loss = 0.
    #     # We assume that bptt and criterion are defined elsewhere.
    #     with torch.no_grad():
    #         for i in tqdm(range(0, data_source.size(0) - 1, bptt)):
    #             data, targets = get_batch(data_source, i)
    #             # Initiate coroutine chain for each batch.
    #             output_flat = await evaluate_partition(data, model[0], next_partition=model[1:])
    #             # Flatten the output for loss calculation
    #             output_flat = output_flat.view(-1, ntokens)
    #             # Move targets to the CPU.
    #             total_loss += len(data) * criterion(output_flat, targets).item()
    #     # Average the loss.
    #     return total_loss / (len(data_source) - 1)

    # # Function to start the evaluation coroutine for the whole dataset.
    # async def evaluate(model, data_source):
    #     ntokens = len(vocab)
    #     # Start evaluating the whole dataset asynchronously.
    #     total_loss = await evaluate_coroutine(model, data_source, ntokens)
    #     return total_loss
        
    print("***** Running evaluation *****")
    print(f"  Num examples = {len(test_data)}")
    
    # test_loss = asyncio.run(evaluate(stages, test_data))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(evaluate(stages, test_data))
    finally:
        loop.close()
    
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))
    # print('=' * 89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--bptt', type=int, default=25, help='batch size')
    parser.add_argument('--emsize', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=4096, help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=12, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=16, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--profile', action='store_true', help='profile training')

    args = parser.parse_args()
    
    if args.profile:
        with torch.cuda.profiler.profile():
            main()
    else:
        main()
