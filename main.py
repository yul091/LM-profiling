import os
import sys
sys.dont_write_bytecode = True
import json
import random
import argparse
import asyncio
from typing import List
from collections import defaultdict
import torch 
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

from utils import get_total_params
from models import Encoder, Decoder, PipelineStage
from producer import Producer
from scheduler import GlobalScheduler
from consumer import Node


class DistributedTransformerPipeline:
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        emsize = args.emsize
        nhid = args.nhid
        nlayers = args.nlayers
        nhead = args.nhead
        dropout = args.dropout
        self.output_dir = args.output_dir
        self.coroutine = args.coroutine
        self.setting = args.setting
        self.profiling = args.profiling
        seed = args.seed
        
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Initialize async components
        self.task_queue = asyncio.Queue()
        self.task_complete_flag = asyncio.Event()
        
        self.vocab = self.producer.vocab
        self.ntokens = len(self.vocab)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])
        node1 = Node(id=1, cuda_devices=[0, 1], criterion=self.criterion)
        node2 = Node(id=2, cuda_devices=[2, 3], criterion=self.criterion)
        self.nodes = [node1, node2]
        
        self.producer = Producer(self.task_queue, args, self.task_complete_flag)
        self.scheduler = GlobalScheduler(
            self.task_queue, 
            self.nodes, 
            self.task_complete_flag,
        )
        
        # Create pipelines for model parallel
        self.timing_infos = []
        self.distributed_stages = []
        self.num_gpus = len(self.nodes[0].devices) # we split pipelines into number of devices in each node
        self.partition_len = ((nlayers - 1) // self.num_gpus) + 1
        print("partition (block) size: {}".format(self.partition_len))
        self.model_kwargs = {
            'ntokens': self.ntokens,
            'emsize': emsize,
            'dropout': dropout,
            'nlayers': nlayers,
            'nhead': nhead,
            'nhid': nhid,
            'partition_len': self.partition_len,
        }
        for idx, node in enumerate(self.nodes):
            init_cuda_id = node.devices[0].cuda_id # the first cuda id of the node
            stages = self.create_pipelines(
                **self.model_kwargs,
                init_gpu_id=init_cuda_id, 
            )
            self.distributed_stages.append(stages)
        
        
    
    def create_pipelines(self, ntokens, emsize, dropout, nlayers, nhead, nhid, partition_len, init_gpu_id=0):
        tmp_list = [Encoder(ntokens, emsize, dropout)]
        stages = []

        # Add all the necessary transformer blocks.
        for i in range(nlayers):
            transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
            if i != 0 and i % (partition_len) == 0:
                # Create a new pipeline stage
                stage_device = i // (partition_len) - 1 + init_gpu_id
                print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device))
                stages.append(PipelineStage(tmp_list, stage_device))
                tmp_list = []
                
            tmp_list.append(transformer_block)

        # Add decoder in the end.
        tmp_list.append(Decoder(ntokens, emsize))
        stages.append(PipelineStage(tmp_list, stage_device + 1))
        print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device + 1))
        print('Total parameters in model: {:,}'.format(get_total_params(nn.Sequential(*stages))))
        return stages
    

    async def main(self):
        
        # Start all components as asyncio tasks
        producer_task = asyncio.create_task(self.producer.produce())
        scheduler_task = asyncio.create_task(self.scheduler.schedule())
        # device_tasks = [asyncio.create_task(device.process_tasks()) for node in nodes for device in node.devices]
        
        consumer_tasks = []
        for idx, node in enumerate(self.nodes):
            stages = self.distributed_stages[idx]
            self.timing_infos.append(defaultdict(list))
            queues = [device.queue for device in node.devices] + [asyncio.Queue()]
            # Start stage coroutines
            stages_coros = [node.coroutine_inference_stage(
                stages[i], 
                queues[i],  # input queue (of data) for stage i
                queues[i + 1],  # output queue (of data) for stage i+1
                device_id=i, 
                timing_info=self.timing_infos[idx],
                next_device_id=i+1 if i < len(stages) - 1 else None,
            ) for i in range(len(stages))]
            
            # Start the consumer coroutine
            consumer_coro = node.compute_loss(
                queues[-1], 
                self.ntokens, 
                len(self.producer.dataset),
            )
            coros = stages_coros + [consumer_coro]
            tasks = [asyncio.create_task(coro) for coro in coros]
            consumer_tasks.extend(tasks)
        
        # Run all tasks until complete
        await asyncio.gather(producer_task, scheduler_task, *consumer_tasks)
        
        
    def save_profiling(self): # After all batches are processed, save the timing information to a file
        os.makedirs(self.output_dir, exist_ok=True)
        execution = 'coroutine' if self.coroutine else 'sync'
        for idx, timing_info in enumerate(self.timing_infos):
            gpus = list(set(int(key.split('_')[0]) for key in timing_info))
            # Remove the first start and end time for each GPU
            for gpu_id in gpus:
                timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start'][1:]
                timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end'][1:]
            stats_f = f'{self.output_dir}/timing_info_{execution}_{self.setting}_node{idx}.json'
            with open(stats_f, 'w') as f:
                json.dump(timing_info, f, indent=4)
    
    
if __name__ == "__main__":
    
    # Test the producer
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], 
                        default='random', help='workload setting')
    parser.add_argument('--output_dir', type=str, default='prof', help='output directory')
    parser.add_argument('--bptt', type=int, default=5, help='batch size')
    parser.add_argument('--emsize', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=4096, help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=6, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=16, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--profiling', action='store_true', help='enable profiling')
    parser.add_argument('--coroutine', action='store_true', help='coroutine inference')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_samples', type=int, default=-1, help='number of samples to profile')
    args = parser.parse_args()
    
    dppl = DistributedTransformerPipeline(args)
    asyncio.run(dppl.main())
    dppl.save_profiling()
    