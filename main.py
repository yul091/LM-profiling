import os
import sys
sys.dont_write_bytecode = True
import json
import time
import math
import random
import argparse
import asyncio
from tqdm import tqdm
from typing import List, Union
from collections import defaultdict

import torch 
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoderLayer

from utils import get_total_params, record_time
from dataset import get_data, SentencePairDataset
from models import Encoder, Decoder, PipelineStage


class Task:
    """
    This class encapsulates a task to be processed by the system.
    query (str): the query batch to be processed
    timestamp (float): the time at which the task was generated
    feedback (str): the feedback for the query (optional) from the user
    """
    def __init__(self, query: Tensor, timestamp: float, feedback: Tensor = None):
        self.query = query
        self.timestamp = timestamp
        self.feedback = feedback
        self.processed = False
        

class DeviceQueue:
    def __init__(self, cuda_id: int):
        self.queue = asyncio.Queue()
        self.cuda_id = cuda_id

    async def add_task(self, task: Task):
        """
        Task can be either a Task object or an integer (task ID).
        """
        # if task is not None:
        #     task.query = task.query.cuda(self.cuda_id) # put query on the device
        await self.queue.put(task)

    def process_task(self, task: Task, timing_info: dict, stage: nn.Module, next_cuda_id: int = None, verbose: bool = False):
        # Inference
        if task.feedback is not None:
            # Record the start time of the stage on this GPU
            record_time(self.cuda_id, 'start', 'forward_loss', timing_info, verbose=verbose)
            output = stage(task.query)
            record_time(self.cuda_id, 'end', 'forward_loss', timing_info, verbose=verbose)
            if next_cuda_id is not None:
                output = output.cuda(next_cuda_id)  # Move output to the next stage's device
            task.query = output
        else:
            with torch.no_grad():
                # Record the start time of the stage on this GPU
                record_time(self.cuda_id, 'start', 'forward', timing_info, verbose=verbose)
                output = stage(task.query)
                record_time(self.cuda_id, 'end', 'forward',  timing_info, verbose=verbose)
                if next_cuda_id is not None:
                    output = output.cuda(next_cuda_id)  # Move output to the next stage's device
                task.query = output
        
        

class Node:
    """
    Suppose each node (server) has 2 GPUs, each having a queue.
    """
    def __init__(
        self, 
        id: int, 
        cuda_devices: List[int], 
        criterion: nn.CrossEntropyLoss, 
        verbose: bool = False,
    ):
        self.id = id
        self.devices = [DeviceQueue(cuda_id) for cuda_id in cuda_devices]
        self.criterion = criterion
        self.verbose = verbose
        
    async def add_task(self, task: Union[Task, int], preloaded_tasks: List[Task] = None):
        if task is None or preloaded_tasks is None:
            await self.devices[0].add_task(task) # always add input data on the first device
        else:
            await self.devices[0].add_task(preloaded_tasks[task]) # always add input data on the first device
        
    async def coroutine_inference_stage(
        self,
        stage: nn.Module, 
        device_queue_in: asyncio.Queue, 
        device_queue_out: asyncio.Queue, 
        device_id: int, 
        timing_info: dict, 
        next_device_id: int = None,
        optimizer: torch.optim.Optimizer = None,
    ):
        while True:
            task: Task = await device_queue_in.get()
            if optimizer is not None:
                optimizer.zero_grad()
            # print(f"[Device {device_id} CUDA {self.devices[device_id].cuda_id}] Task received at time {time.time()}")  # Debug
            if task is None:  # None is sent as a signal to shut down
                print("[node {}] Device {} all task finished !".format(self.id, device_id))
                await device_queue_out.put(None)
                break
            self.devices[device_id].process_task(
                task=task, 
                timing_info=timing_info, 
                stage=stage, 
                next_cuda_id=self.devices[next_device_id].cuda_id if next_device_id is not None else None,
                verbose=self.verbose,
            )
            await device_queue_out.put(task)


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
        bptt = args.bptt
        n_samples = args.n_samples
        self.rate_lambda = args.rate_lambda
        lr = args.lr
        self.verbose = args.verbose
        self.workload = args.workload
        self.use_preload = args.use_preload
        self.retraining_rate = args.retraining_rate
        
        # Reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Initialize async components
        self.task_queue = asyncio.Queue()
        train_data, val_data, test_data, self.vocab = get_data(setting=args.setting)
        self.ntokens = len(self.vocab)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])
        num_gpus = torch.cuda.device_count()
        half_gpus = num_gpus // 2
        node1 = Node(
            id=1, 
            cuda_devices=list(range(half_gpus)), 
            criterion=self.criterion,
            verbose=self.verbose,
        )
        node2 = Node(
            id=2, 
            cuda_devices=list(range(half_gpus, num_gpus)), 
            criterion=self.criterion,
            verbose=self.verbose,
        )
        self.nodes = [node1, node2]
        
        # Create producer
        def collate_batch(batch):
            # 'batch' is a list of tuples with (sequence, target)
            batch_data, batch_target = zip(*batch)
            combined_list = batch_data + batch_target
            # Dynamically pad the batch
            padded = pad_sequence(
                combined_list, 
                batch_first=True, 
                padding_value=self.vocab['<pad>'],
            )
            padded_data = padded[:len(batch_data)]
            padded_target = padded[len(batch_data):]
            return padded_data, padded_target.view(-1)
        
        self.dataset = SentencePairDataset(test_data)
        if n_samples > 0:
            self.dataset = Subset(
                self.dataset, 
                random.sample(range(len(self.dataset)), n_samples),
            )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=bptt, 
            collate_fn=collate_batch,
        )
        self.total_tasks = len(self.dataloader)
        self.task_completed = 0
        
        # Create preloaded data
        if self.use_preload:
            print("Using preloaded data")
            self.preloaded_tasks = defaultdict(list)
            for node in self.nodes:
                for batch in self.dataloader:
                    # 10% of the time, produce a task with feedback
                    if random.random() < self.retraining_rate:
                        task = Task(
                            query=batch[0].cuda(node.devices[0].cuda_id), 
                            timestamp=time.time(), 
                            feedback=batch[1].cuda(node.devices[-1].cuda_id), 
                        )
                    else:
                        task = Task(
                            query=batch[0].cuda(node.devices[0].cuda_id),
                            timestamp=time.time(),
                        )
                    self.preloaded_tasks[node.id].append(task)
                    
        # Create pipelines for model parallel
        self.timing_infos = []
        self.distributed_stages = []
        self.optimizers = []
        self.lr_schedulers = []
        self.num_gpus = len(self.nodes[0].devices) 
        self.partition_len = ((nlayers - 1) // self.num_gpus) + 1
        print("partition (block) size: {}".format(self.partition_len))
        model_kwargs = {
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
                **model_kwargs,
                init_gpu_id=init_cuda_id, 
            )
            self.distributed_stages.append(stages)
            optimizer = torch.optim.SGD(nn.Sequential(*stages).parameters(), lr=lr)
            self.optimizers.append(optimizer)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
            self.lr_schedulers.append(scheduler)
        
    
    def create_pipelines(
        self, 
        ntokens: int, 
        emsize: int, 
        dropout: float, 
        nlayers: int, 
        nhead: int, 
        nhid: int, 
        partition_len: int, 
        init_gpu_id: int = 0,
    ) -> List[PipelineStage]:
        stages = []
        tmp_list = [Encoder(ntokens, emsize, dropout)]
        # Add all the necessary transformer blocks
        for i in range(nlayers):
            transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
            if i != 0 and i % (partition_len) == 0:
                # Create a new pipeline stage
                stage_device = i // (partition_len) - 1 + init_gpu_id
                print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device))
                stages.append(PipelineStage(tmp_list, stage_device))
                tmp_list = []
                
            tmp_list.append(transformer_block)
        # Add decoder in the end
        tmp_list.append(Decoder(ntokens, emsize))
        stages.append(PipelineStage(tmp_list, stage_device + 1))
        print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device + 1))
        print('Total parameters in model: {:,}'.format(get_total_params(nn.Sequential(*stages))))
        return stages
    
    async def produce(self):
        # Produce using the dataset
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            # print(f"query shape: {batch[0].shape}, target shape: {batch[1].shape}")
            if self.workload == 'poisson':
                await asyncio.sleep(random.expovariate(self.rate_lambda))
            elif self.workload == 'all':
                await asyncio.sleep(0)
            else:
                raise ValueError(f"Invalid workload type: {self.workload}")
            # 10% of the time, produce a task with feedback
            if self.use_preload: 
                # Essentially, we are using preloaded data (task ID)
                await self.task_queue.put(i)
            else:  
                if random.random() < self.retraining_rate:
                    task = Task(query=batch[0], timestamp=time.time(), feedback=batch[1])
                else:
                    task = Task(query=batch[0], timestamp=time.time())
                await self.task_queue.put(task)
            
        await self.task_queue.put(None)  # Signal the end of the dataset
        print("Producer finished producing tasks")
    
    
    async def schedule(self):
        while True:
            task = await self.task_queue.get()
            if task is None:
                print("Scheduler finished scheduling tasks")
                # Add none to both nodes to signal the end of the dataset
                for node in self.nodes:
                    await node.add_task(None)
                break
            node = random.choice(self.nodes)
            if self.use_preload:
                await node.add_task(task, self.preloaded_tasks[node.id])
            else:
                await node.add_task(task)
            # print(f"Task scheduled to node {node.id} at time {time.time()}")
            # print(f"Task queue size: {self.task_queue.qsize()}")
            
            
    async def compute_loss(
        self, 
        queue: asyncio.Queue, 
        stages: List[PipelineStage], 
        optimizer: torch.optim.Optimizer,
        node: Node,
        timing_info: dict = None,
    ):
        total_loss = 0.
        while self.task_completed < self.total_tasks:
            # print(f"{self.task_completed}/{self.total_tasks} tasks completed !!!")
            task: Task = await queue.get() # (B X T X C) and (B*T)
            if task is None:
                break
            if task.feedback is None:
                self.task_completed += 1
                continue
            output_flat = task.query.contiguous().view(-1, self.ntokens) # (B*T) X C
            batch_loss = self.criterion(output_flat, task.feedback)
            if self.verbose:
                print(f"[node {node.id}] batch loss: {batch_loss.item()}")
            total_loss += task.query.size(0) * batch_loss.item() 
            # print(f"query shape: {task.query.shape}, target shape: {task.feedback.shape}")
            # Backpropagate the loss
            record_time(node.devices[-1].cuda_id, 'start', 'backward', timing_info, verbose=self.verbose)
            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(nn.Sequential(*stages).parameters(), 0.5)
            optimizer.step()
            record_time(node.devices[-1].cuda_id, 'end', 'backward', timing_info, verbose=self.verbose)
            self.task_completed += 1
        
        print(f"[node {node.id}] total loss: {total_loss/len(self.dataset)}, perplexity: {math.exp(total_loss/len(self.dataset))}")
        # return total_loss / len(self.dataset)
    

    async def main(self):
        
        execution_coros = []
        for idx, node in enumerate(self.nodes):
            stages = self.distributed_stages[idx]
            self.timing_infos.append(defaultdict(list))
            queues = [device.queue for device in node.devices] + [asyncio.Queue()]
            optimizer = self.optimizers[idx]

            # Start stage coroutines
            stages_coros = [node.coroutine_inference_stage(
                stages[i], 
                queues[i],  # input queue (of data) for stage i
                queues[i + 1],  # output queue (of data) for stage i+1
                device_id=i, 
                timing_info=self.timing_infos[idx],
                next_device_id=i+1 if i < len(stages) - 1 else None,
                optimizer=optimizer,
            ) for i in range(len(stages))]
            execution_coros.extend(stages_coros)
            
            # Start the consumer coroutine
            if self.retraining_rate > 0:
                consumer_coro = self.compute_loss(
                    queue=queues[-1],
                    stages=stages, 
                    optimizer=optimizer,
                    node=node,
                    timing_info=self.timing_infos[idx],
                )
                execution_coros.append(consumer_coro)
        
        # Run all tasks until complete
        producer_coro = self.produce()
        scheduler_coro = self.schedule()
        coros = [producer_coro] + [scheduler_coro] + execution_coros
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
        
    def save_profiling(self):
        os.makedirs(self.output_dir, exist_ok=True)
        execution = 'coroutine' if self.coroutine else 'sync'
        for idx, timing_info in enumerate(self.timing_infos):
            # Remove the first start and end time for each GPU
            gpus = list(set(int(key.split('_')[0]) for key in timing_info))
            for gpu_id in gpus:
                timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start']
                timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end']
            stats_f = f'{self.output_dir}/timing_info_{execution}_{self.setting}_{self.workload}_node{idx}.json'
            with open(stats_f, 'w') as f:
                json.dump(timing_info, f, indent=4)
    
    
if __name__ == "__main__":
    
    # Test the producer
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], 
                        default='random', help='workload setting')
    parser.add_argument('--output_dir', type=str, default='prof', help='output directory')
    parser.add_argument('--bptt', type=int, default=10, help='batch size')
    parser.add_argument('--emsize', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=4096, help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=12, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=16, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--profiling', action='store_true', help='enable profiling')
    parser.add_argument('--coroutine', action='store_true', help='coroutine inference')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_samples', type=int, default=-1, help='number of samples to profile')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--workload', type=str, choices=['poisson', 'all'], default='poisson', help='workload type')
    parser.add_argument('--use_preload', action='store_true', help='use preloaded data (already in the first GPU of each node)')
    parser.add_argument('--retraining_rate', type=float, default=0.1, help='retraining rate')
    args = parser.parse_args()
    
    dppl = DistributedTransformerPipeline(args)
    asyncio.run(dppl.main())
    dppl.save_profiling()
    