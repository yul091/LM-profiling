import os
import sys
sys.dont_write_bytecode = True
import math
import time
import json
import argparse
from tqdm import tqdm
import asyncio
from collections import defaultdict
from typing import Dict, List

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
    
def record_time(device: int, event_type: str, timing_info: Dict[str, List[float]]):
    # event_type can be 'start' or 'end'
    timing_info[f"{device}_{event_type}"].append(time.time())
    
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
    output_dir = args.output_dir
    coroutine = args.coroutine
    
    train_data, val_data, test_data, vocab = get_data()
    ntokens = len(vocab) # the size of vocabulary
    
    def get_batch(source, i):
        # Source (N X T) where T is the chunk size
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # (B X T) where B is the batch size
        target = source[i+1:i+1+seq_len].view(-1) # (B*T) the corresponding next sentences
        # Need batch dimension first for pipeline parallelism.
        return data.t(), target

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
    stages.append(PipelineStage(tmp_list, stage_device + 1))
    print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device + 1))
    print ('Total parameters in model: {:,}'.format(get_total_params(torch.nn.Sequential(*stages))))

    # Run the model
    criterion = nn.CrossEntropyLoss()
    # Dictionary to hold all timing information
    timing_info = defaultdict(list)
    
    async def producer(queue, data_source, bptt):
        # Evaluate only for 50 batches to keep script execution time low.
        # nbatches = min(50 * bptt, data_source.size(0) - 1)
        nbatches = data_source.size(0) - 1
        for i in tqdm(range(0, nbatches, bptt)):
            data, targets = get_batch(data_source, i)
            await queue.put((data, targets))
            await asyncio.sleep(0)  # Simulates a long-running task
        await queue.put((None, None))  # Signal the end of the dataset
    
    async def coroutine_evaluate_stage(stage, queue_in, queue_out, device, next_device=None):
        while True:
            data, targets = await queue_in.get()
            if data is None:  # None is sent as a signal to shut down.
                await queue_out.put((None, None))
                break
            with torch.no_grad():
                # Record the start time of the stage on this GPU
                record_time(device, 'start', timing_info)
                output = stage(data)
                record_time(device, 'end', timing_info)
                if next_device:
                    output = output.to(next_device)  # Move output to the next stage's device.
            await queue_out.put((output, targets))
            
    async def consumer(queue, ntokens):
        total_loss = 0.
        while True:
            output, targets = await queue.get() # (B X T X C) and (B*T)
            if output is None:
                break
            output_flat = output.contiguous().view(-1, ntokens) # (B*T) X C
            total_loss += output.size(0) * criterion(output_flat, targets.to(output.device)).item() 
        return total_loss / (len(test_data) - 1)
    
    async def coroutine_evaluate(stages, data_source):
        queues = [asyncio.Queue() for _ in range(len(stages) + 1)]
        
        # Start the producer coroutine
        producer_coro = producer(queues[0], data_source, bptt)

        # Start stage coroutines
        stages_coros = [coroutine_evaluate_stage(
            stages[i], 
            queues[i],  # input queue (of data) for stage i
            queues[i + 1],  # output queue (of data) for stage i+1
            device=i, 
            next_device=i+1 if i < len(stages) - 1 else None,
        ) for i in range(len(stages))]

        # Start the consumer coroutine
        consumer_coro = consumer(queues[-1], len(vocab))

        # Run the coroutines
        coros = [producer_coro] + stages_coros + [consumer_coro]
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
        
        # # Get the result from the consumer
        # loss = await consumer_coro
        # Instead of awaiting the consumer coroutine again, get the result from the Task object
        consumer_task = tasks[-1]  # This assumes that the consumer task is the last in the list
        if consumer_task.done():
            loss = consumer_task.result()
        else:
            loss = None  # or handle accordingly
        print(f"Test Loss: {loss}, perplexity: {math.exp(loss)}")
        
    def evaluate(stages, data_source):
        total_loss = 0.
        ntokens = len(vocab)
        # Evaluate only for 50 batches to keep script execution time low.
        # nbatches = min(50 * bptt, data_source.size(0) - 1)
        nbatches = data_source.size(0) - 1
        with torch.no_grad():
            for i in tqdm(range(0, nbatches, bptt)):
                data, targets = get_batch(data_source, i)
                output = data
                for i, stage in enumerate(stages):
                    record_time(i, 'start', timing_info)
                    stage.eval()
                    output = output.cuda(i)
                    output = stage(output)
                    record_time(i, 'end', timing_info)

                output_flat = output.contiguous().view(-1, ntokens)
                # Need to move targets to the device where the output of the
                # pipeline resides.
                total_loss += len(data) * criterion(output_flat, targets.to(output.device)).item()
        return total_loss / (len(data_source) - 1)
        
    print("***** Running evaluation *****")
    print(f"  Num examples = {len(test_data)}")
    
    if coroutine:
        print("  Running coroutine inference ...")
        asyncio.run(coroutine_evaluate(stages, test_data))
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # try:
        #     loop.run_until_complete(evaluate(stages, test_data))
        # finally:
        #     loop.close()
    else:
        print("  Running synchronous inference ...")
        test_loss = evaluate(stages, test_data)
        print('test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
        
        
    # After all batches are processed, save the timing information to a file
    os.makedirs(output_dir, exist_ok=True)
    stats_f = f'{output_dir}/timing_info_coroutine.json' if coroutine else f'{output_dir}/timing_info.json'
    with open(stats_f, 'w') as f:
        json.dump(timing_info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--output_dir', type=str, default='prof', help='output directory')
    parser.add_argument('--bptt', type=int, default=25, help='batch size')
    parser.add_argument('--emsize', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=4096, help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=12, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=16, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--coroutine', action='store_true', help='coroutine inference')

    args = parser.parse_args()
    
    main()
