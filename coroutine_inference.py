import os
import sys
sys.dont_write_bytecode = True
import math
import time
import json
import wandb
import random
import argparse
from tqdm import tqdm
import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from dataset import get_data, SentencePairDataset
from models import Encoder, Decoder, PipelineStage
from utils import record_time, get_total_params

if torch.cuda.device_count() < 2:
    print('Need at least two GPU devices for this tutorial')
    sys.exit(0)


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
    setting = args.setting
    profiling = args.profiling
    seed = args.seed
    n_samples = args.n_samples
    
    train_data, val_data, test_data, vocab = get_data(setting=setting)
    
    def collate_batch(batch):
        # 'batch' is a list of tuples with (sequence, target)
        batch_data, batch_target = zip(*batch)
        combined_list = batch_data + batch_target
        # Dynamically pad the batch
        padded = pad_sequence(combined_list, batch_first=True, padding_value=vocab['<pad>'])
        padded_data = padded[:len(batch_data)]
        padded_target = padded[len(batch_data):]
        return padded_data, padded_target.view(-1)
    
    # def get_batch(source, i):
    #     # Source (N X T) where T is the chunk size
    #     seq_len = min(bptt, len(source) - 1 - i)
    #     data = source[i:i+seq_len] # (B X T) where B is the batch size
    #     target = source[i+1:i+1+seq_len].view(-1) # (B*T) the corresponding next sentences
    #     # Need batch dimension first for pipeline parallelism.
    #     return data.t(), target
    
    test_dataset = SentencePairDataset(test_data, setting='identical')
    if n_samples > 0:
        test_dataset = Subset(test_dataset, random.sample(range(len(test_dataset)), n_samples))
    ntokens = len(vocab) # the size of vocabulary
    test_loader = DataLoader(
        test_dataset, 
        batch_size=bptt, 
        collate_fn=collate_batch,
        shuffle=False,
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
        
    # Preload dataset to GPU memory
    preloaded_tasks = []
    for i, (data, targets) in enumerate(test_loader):
        # 10% of the time, produce a task with feedback
        preloaded_tasks.append((data.cuda(0), targets.cuda(num_gpus - 1)))

    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize))
    stages.append(PipelineStage(tmp_list, stage_device + 1))
    print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device + 1))
    print ('Total parameters in model: {:,}'.format(get_total_params(torch.nn.Sequential(*stages))))

    # Run the model
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    # Dictionary to hold all timing information
    timing_info = defaultdict(list)
    
    async def producer(queue: asyncio.Queue, data_source: List[Tuple[Tensor, Tensor]]):
        for i, (data, targets) in tqdm(enumerate(data_source), total=len(data_source)):
            # data, targets = get_batch(data_source, i)
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
                record_time(device, 'start', 'forward', timing_info)
                output = stage(data)
                record_time(device, 'end', 'forward', timing_info)
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
            total_loss += output.size(0) * criterion(output_flat, targets).item() 
        return total_loss / (len(test_data) - 1)
    
    async def coroutine_evaluate(stages, data_source):
        queues = [asyncio.Queue() for _ in range(len(stages) + 1)]
        
        # Start the producer coroutine
        producer_coro = producer(queues[0], data_source)

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
        
        # Instead of awaiting the consumer coroutine again, get the result from the Task object
        consumer_task = tasks[-1]  # This assumes that the consumer task is the last in the list
        if consumer_task.done():
            loss = consumer_task.result()
        else:
            loss = None  # or handle accordingly
        print(f"Test Loss: {loss}, perplexity: {math.exp(loss)}")
        # wandb.log({"test_loss": loss, "test_ppl": math.exp(loss)})
        
    def evaluate(stages: List[nn.Sequential], data_source: DataLoader):
        total_loss = 0.
        ntokens = len(vocab)
        with torch.no_grad():
            # for i in tqdm(range(0, nbatches, bptt)):
            for i, (data, targets) in tqdm(enumerate(data_source), total=len(data_source)):
                # data, targets = get_batch(data_source, i) # (B X T) and (B*T)
                data = data.cuda(0)
                output = data 
                for j, stage in enumerate(stages):
                    record_time(j, 'start', 'forward', timing_info)
                    stage.eval()
                    output = output.cuda(j)
                    output = stage(output) # (B X T X C)
                    record_time(j, 'end', 'forward', timing_info)

                output_flat = output.contiguous().view(-1, ntokens) # (B*T) X C
                # Need to move targets to the device where the output of the pipeline resides.
                batch_loss = criterion(output_flat, targets.to(output.device)).item()
                # print(f"Batch {i} loss: {batch_loss}")
                total_loss += len(data) * batch_loss
                
        return total_loss / (len(test_data) - 1)
        
    print("***** Running evaluation *****")
    print(f"  Num examples = {len(test_data)}")
    print(f"  Workload setting = {setting}")
    
    random.seed(seed)
    execution = 'coroutine' if coroutine else 'sync'
    
    if profiling:
        wandb.login()
        wandb.init(
            project="coroutine_inference",
            name=f"{execution}_{setting}_workload",
            settings=wandb.Settings(
                _stats_sample_rate_seconds=2, # high resolution system metrics
                _stats_samples_to_average=1,
            ),
            config={
                "bptt": bptt,
                "emsize": emsize,
                "nhid": nhid,
                "nlayers": nlayers,
                "nhead": nhead,
                "dropout": dropout,
                "coroutine": coroutine,
            }
        )
    
    if coroutine:
        print("  Running coroutine inference ...")
        # asyncio.run(coroutine_evaluate(stages, test_data))
        # asyncio.run(coroutine_evaluate(stages, test_loader))
        asyncio.run(coroutine_evaluate(stages, preloaded_tasks))
    else:
        print("  Running synchronous inference ...")
        # test_loss = evaluate(stages, test_data)
        test_loss = evaluate(stages, test_loader)
        print('test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
        # wandb.log({"test_loss": test_loss, "test_ppl": math.exp(test_loss)})
        
    # After all batches are processed, save the timing information to a file
    # Remove the first start and end time for each GPU
    for gpu_id in range(num_gpus):
        timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start'][1:]
        timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end'][1:]
    
    os.makedirs(output_dir, exist_ok=True)
    stats_f = f'{output_dir}/timing_info_{execution}_{setting}.json'
    with open(stats_f, 'w') as f:
        json.dump(timing_info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--output_dir', type=str, default='prof', help='output directory')
    parser.add_argument('--bptt', type=int, default=5, help='batch size')
    parser.add_argument('--emsize', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=4096, help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=12, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=16, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--profiling', action='store_true', help='enable profiling')
    parser.add_argument('--coroutine', action='store_true', help='coroutine inference')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], default='identical', help='workload setting')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_samples', type=int, default=-1, help='number of samples to profile')

    args = parser.parse_args()
    
    main()
