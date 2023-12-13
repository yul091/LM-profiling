import os
import sys
sys.dont_write_bytecode = True
import json
import time
import queue
import random
import argparse
from tqdm import tqdm
from typing import List, Dict, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader, Subset

from utils import get_total_params, record_time
from dataset import get_data, SentencePairDataset
from models import Encoder, Decoder, PipelineStage

# Define the job class
class Job:
    def __init__(
        self, 
        task_id: int,
        data: Tensor = None, 
        in_degree: int = 1,
    ):
        self.task_id = task_id
        self.cuda_id = None
        self.isLast = False
        self.val = data
        self.in_degree = in_degree


# Define the task class
class Task:
    def __init__(
        self, 
        task_id: int, 
        num_jobs: int, 
        query: Tensor, 
        feedback: Tensor = None,
    ):
        self.task_id = task_id
        self.num_jobs = num_jobs
        self.query = query
        self.jobs = [Job(task_id, query, in_degree=0) if i == 0 else Job(task_id, in_degree=1) 
                     for i in range(num_jobs)]
        self.feedback = feedback
        
    def get_job(self, device: int) -> Job:
        return self.jobs[device % self.num_jobs]
    
    
# Define the device class
class Device:
    def __init__(self, cuda_id: int):
        self.cuda_id = cuda_id
        self.isGPUavailable = True
        self.queue = queue.Queue()
        
        
# Define the Node class
class Node:
    def __init__(self, node_id: int, num_devices: int):
        self.node_id = node_id
        self.num_devices = num_devices
        self.devices = [Device(node_id * num_devices + device) for device in range(num_devices)]
        self.queue = queue.Queue()
        
    def get_device_from_job(self, job: Job) -> Device:
        return self.devices[job.cuda_id % self.num_devices]
    
    def get_next_device_from_job(self, job: Job) -> Device:
        return self.devices[(job.cuda_id + 1) % self.num_devices]
        
        
# Define the Pipeline class
class DistributedLLM:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed = args.seed
        self.bptt = args.bptt
        self.output_dir = args.output_dir
        self.setting = args.setting
        self.n_samples = args.n_samples
        self.rate_lambda = args.rate_lambda
        self.lr = args.lr
        
        # Reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        # System arguments
        num_gpus = torch.cuda.device_count()
        self.num_nodes = args.num_nodes
        self.num_devices_per_node = num_gpus // self.num_nodes
        self.nodes = [Node(id, self.num_devices_per_node) for id in range(self.num_nodes)]
        
        # Dataset arguments
        self.verbose = args.verbose
        self.workload = args.workload
        self.use_preload = args.use_preload
        self.retraining_rate = args.retraining_rate
        _, _, test_data, self.vocab = get_data(
            setting=self.setting,
            bptt=self.bptt,
        )
        self.get_dataloader(test_data)
        self.total_tasks = len(self.dataloader)
        self.ntokens = len(self.vocab)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])
        self.preloaded_tasks = self.get_preloaded_dataset()
        self.completed_tasks = 0
        
        # Model arguments
        self.distributed_stages = []
        self.optimizers = []
        self.lr_schedulers = []
        self.partition_len = ((args.nlayers - 1) // self.num_devices_per_node) + 1
        print("partition (block) size: {}".format(self.partition_len))
        model_kwargs = {
            'ntokens': self.ntokens,
            'emsize': args.emsize,
            'nhid': args.nhid,
            'nlayers': args.nlayers,
            'nhead': args.nhead,
            'dropout': args.dropout,
            'partition_len': self.partition_len,
        }
        for node in self.nodes:
            init_cuda_id = node.devices[0].cuda_id # the first cuda id of the node
            stages = self.create_pipelines(
                **model_kwargs,
                init_gpu_id=init_cuda_id, 
            )
            self.distributed_stages.append(stages)
            optimizer = torch.optim.SGD(nn.Sequential(*stages).parameters(), lr=self.lr)
            self.optimizers.append(optimizer)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
            self.lr_schedulers.append(scheduler)
        
        
        
    def get_dataloader(self, dataset: List[Tensor]):
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
        
        self.dataset = SentencePairDataset(dataset, setting=self.setting)
        if self.n_samples > 0:
            if self.setting == 'random':
                indices = random.sample(range(len(self.dataset)), self.n_samples)
            elif self.setting == 'variant':
                indices = list(range(self.n_samples))
            self.dataset = Subset(
                self.dataset, 
                indices,
            )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.bptt, 
            collate_fn=collate_batch,
            shuffle=False,
        )
        
        
    def get_preloaded_dataset(self) -> Dict[int, List[Task]]:
        print("Using preloaded data ...")
        preloaded_tasks = defaultdict(list)
        for node in self.nodes:
            if self.setting == 'random':
                for i, batch in enumerate(self.dataloader):
                    # 10% of the time, produce a task with feedback
                    if random.random() < self.retraining_rate:
                        task = Task(
                            task_id=i,
                            num_jobs=self.num_devices_per_node,
                            query=batch[0].cuda(node.devices[0].cuda_id), 
                            feedback=batch[1].cuda(node.devices[-1].cuda_id), 
                        )
                    else:
                        task = Task(
                            task_id=i,
                            num_jobs=self.num_devices_per_node,
                            query=batch[0].cuda(node.devices[0].cuda_id),
                        )
                    preloaded_tasks[node.node_id].append(task)
                    
            elif self.setting == 'variant':
                for i, batch in enumerate(self.dataloader):
                    # Odd batches are short, better utilize the bubble for retraining
                    if i % 2 == 0 and random.random() < 2 * self.retraining_rate:
                        task = Task(
                            task_id=i,
                            num_jobs=self.num_devices_per_node,
                            query=batch[0].cuda(node.devices[0].cuda_id), 
                            feedback=batch[1].cuda(node.devices[-1].cuda_id), 
                        )
                    else:
                        task = Task(
                            task_id=i,
                            num_jobs=self.num_devices_per_node,
                            query=batch[0].cuda(node.devices[0].cuda_id),
                        )
                    preloaded_tasks[node.node_id].append(task)
        return preloaded_tasks

        
    def create_pipelines(self, ntokens, emsize, dropout, nlayers, nhead, nhid, partition_len, init_gpu_id=0) -> List[PipelineStage]:
        if partition_len == nlayers:
            stages = [nn.Sequential(
                Encoder(ntokens, emsize, dropout),
                *[TransformerEncoderLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers)],
                Decoder(ntokens, emsize),
            ).to(init_gpu_id)]
            print("Put stage {} on device {}".format([stages[0].__class__.__name__], init_gpu_id))
        else:
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
    
    
    def producer(self, taskQueue: queue.Queue):
        # Produce using the dataset
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            # print(f"query shape: {batch[0].shape}, target shape: {batch[1].shape}")
            if self.workload == 'poisson':
                time.sleep(random.expovariate(self.rate_lambda))
            elif self.workload == 'all':
                time.sleep(0)
            else:
                raise ValueError(f"Invalid workload type: {self.workload}")
            # 10% of the time, produce a task with feedback
            print("Producing task {} with length {}".format(i, batch[0].size(1)))
            # Essentially, we are using preloaded data (task ID)
            taskQueue.put(i)
            
        taskQueue.put(None)  # Signal the end of the dataset
        print("Producer finished producing tasks")
        
    
    def globalScheduler(self, taskQueue: queue.Queue, nodes: List[Node]):
        while True:
            # Global scheduler
            task: int = taskQueue.get() # ID
            if task is None:
                print("Scheduler finished scheduling tasks")
                for node in nodes:
                    node.queue.put(None)
                break
            node_id = random.randint(0, self.num_nodes - 1)
            # Each node queue store task IDs
            nodes[node_id].queue.put(task)
            print("Global scheduler scheduled task {} to node {}".format(task, node_id))
            
            
    def deviceScheduler(self, node: Node, job_batch: Dict[int, List[Job]]):
        # Device scheduler
        while True:
            for i, device in enumerate(node.devices):
                task: int = device.queue.get() # ID
                if task is None:
                    print(f"Device scheduler finished scheduling tasks for node {node.node_id}")
                    break
                # Define job class
                job = self.preloaded_tasks[node.node_id][task].get_job(i) # Job
                job.cuda_id = device.cuda_id
                job.isLast = i == len(node.devices) - 1
                # Put the job into the job batch for execution
                if device.isGPUavailable and job.in_degree == 0:
                    job_batch[device.cuda_id].append(job)
                else:
                    # Otherwise, put the task ID back to the queue
                    device.queue.put(task)
                    
            if task is None:
                break
            
            
    def job_inference(self, node: Node, job_batch: Dict[int, List[Job]], stage: PipelineStage, timing_info: dict):
        while self.completed_tasks < self.total_tasks: 
            print("Node {} - completed {} tasks".format(node.node_id, self.completed_tasks))
            if not job_batch[stage.device]: # empty execution batch
                continue
            job = job_batch[stage.device].pop(0)
            input = job.val.cuda(job.cuda_id, non_blocking=True)
            
            device = node.get_device_from_job(job)
            record_time(job.cuda_id, 'start', 'forward', timing_info)
            device.isGPUavailable = False # disable GPU availability
            output = stage(input)
            device.isGPUavailable = True # enable GPU availability
            record_time(job.cuda_id, 'end', 'forward', timing_info)
            
            if job.isLast:
                self.completed_tasks += 1
            else:
                # Update the task: the in-degree of the next job is 0, the val is the output
                task = self.preloaded_tasks[node.node_id][job.task_id]
                # next_job = task.get_job(job.cuda_id + 1)
                # next_job.in_degree, next_job.val = 0, output
                task.jobs[(job.cuda_id + 1) % self.num_devices_per_node].val = output
                task.jobs[(job.cuda_id + 1) % self.num_devices_per_node].in_degree = 0
            
            
    def consumer(self, node: Node, stages: List[PipelineStage], job_batch: Dict[int, List[Task]], timing_info: dict):
        task = node.queue.get() # ID
        # Put the task ID into each device queue
        for i, device in enumerate(node.devices):
            device.queue.put(task)
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            while True: 
                future1 = executor.submit(
                    self.deviceScheduler,
                    node, 
                    job_batch,
                )
                for stage in stages:
                    future = executor.submit(
                        self.job_inference,
                        node, 
                        job_batch,
                        stage,
                        timing_info,
                    )
                
                # # Wait for all jobs to finish
                # for future in future2:
                #     future.result()
                # # Wait for the device scheduler to finish
                # future1.result()
                
                # Check if the task is finished
                if self.completed_tasks == self.total_tasks:
                    break
        
            
    def main(self):
        self.timing_infos = [defaultdict(list) for _ in range(self.num_nodes)]
        self.job_batches = [defaultdict(list) for _ in range(self.num_nodes)]
        
        with ThreadPoolExecutor(max_workers=self.num_nodes+2) as executor:
            # Create the producer thread
            taskQueue = queue.Queue()
            # producerThread = threading.Thread(target=self.producer, args=(taskQueue,))
            # producerThread.start()
            future1 = executor.submit(self.producer, taskQueue)
            
            # Create the global scheduler thread
            # globalSchedulerThread = threading.Thread(target=self.globalScheduler, args=(taskQueue, self.nodes))
            # globalSchedulerThread.start()
            future2 = executor.submit(self.globalScheduler, taskQueue, self.nodes)
            
            for i, node in enumerate(self.nodes):
                future = executor.submit(
                    self.consumer, 
                    node, 
                    self.distributed_stages[i], 
                    self.job_batches[i],
                    self.timing_infos[i],
                )
        
        # # Create the node threads
        # nodeThreads = []
        # for node in self.nodes:
        #     nodeThread = threading.Thread(target=self.node, args=(node,))
        #     nodeThreads.append(nodeThread)
        #     nodeThread.start()
            
        # # Wait for all threads to finish
        # producerThread.join()
        # globalSchedulerThread.join()
        # for nodeThread in nodeThreads:
        #     nodeThread.join()
            
        # # Print the timing info
        # for node in self.nodes:
        #     print(f"Node {node.node_id} timing info: {node.timing_info}")
            
        # # Print the total time
        # total_time = 0
        # for node in self.nodes:
        #     total_time += node.timing_info['total']
        # print(f"Total time: {total_time}")
        
    def save_profiling(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for idx, timing_info in enumerate(self.timing_infos):
            # Remove the first start and end time for each GPU
            gpus = list(set(int(key.split('_')[0]) for key in timing_info))
            for gpu_id in gpus:
                timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start']
                timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end']
            stats_f = f'{self.output_dir}/timing_info_coroutine_{self.setting}_{self.workload}_{self.retraining_rate}_node{idx}.json'
            with open(stats_f, 'w') as f:
                json.dump(timing_info, f, indent=4)
    
    
    
if __name__ == "__main__":
    
    # Test the producer
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'variant'], 
                        default='random', help='workload setting')
    parser.add_argument('--output_dir', type=str, default='prof', help='output directory')
    parser.add_argument('--num_nodes', type=int, default=2, help='number of nodes for distributed systems')
    parser.add_argument('--bptt', type=int, default=10, help='batch size')
    parser.add_argument('--emsize', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--nhid', type=int, default=4096, help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=12, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=16, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_samples', type=int, default=-1, help='number of samples to profile')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--workload', type=str, choices=['poisson', 'all'], default='poisson', help='workload type')
    parser.add_argument('--use_preload', action='store_true', help='use preloaded data (already in the first GPU of each node)')
    parser.add_argument('--retraining_rate', type=float, default=0.1, help='retraining rate')
    args = parser.parse_args()
    
    torch.autograd.set_detect_anomaly(True)
    
    dppl = DistributedLLM(args)
    dppl.main()
    dppl.save_profiling()