import os
import sys
sys.dont_write_bytecode = True
import time
import json
import queue
import random
import argparse
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Optional
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor

from utils import record_time
from models import PipelineStage, get_transformer_stages
from dataset import get_data, SentencePairDataset



class Node:
    def __init__(
        self, 
        node_id: int, 
        num_gpus_per_node: int, 
        init_device: Optional[int] = 0,
    ):
        self.node_id = node_id
        self.num_gpus_per_node = num_gpus_per_node
        self.device_queues = [queue.Queue() for _ in range(num_gpus_per_node)]
        self.init_device = init_device
        self.last_device = init_device + num_gpus_per_node - 1
        
        
class Task:
    def __init__(
        self, 
        task_id: int, 
        query: Tensor, 
        feedback: Optional[Tensor] = None, 
        node_id: Optional[int] = None, 
        num_gpus_per_node: Optional[int] = None,
        require_training: Optional[bool] = None,
    ):
        self.task_id = task_id
        self.query = query
        num_gpus_per_node = num_gpus_per_node if num_gpus_per_node is not None else 1
        self.hiddens = [query] + [None for _ in range(num_gpus_per_node - 1)]
        self.feedback = feedback
        self.node_id = node_id if node_id is not None else 0
        self.require_training = False if require_training is None else True
        

class DistributedTransformer:
    
    def __init__(self, args: argparse.Namespace):
        n_samples = args.n_samples
        self.setting = args.setting
        self.num_nodes = args.num_nodes
        nlayers = args.nlayers
        emsize = args.emsize
        nhead = args.nhead
        nhid = args.nhid
        dropout = args.dropout
        batch_size = args.batch_size
        block_size = args.block_size
        self.rate_lambda = args.rate_lambda
        self.output_dir = args.output_dir
        self.workload = args.workload
        self.retraining_rate = args.retraining_rate
        self.num_gpus_per_node = torch.cuda.device_count() // self.num_nodes
        self.distributed_nodes = [
            Node(i, self.num_gpus_per_node, i * self.num_gpus_per_node) 
            for i in range(self.num_nodes)
        ]
        self.timing_infos = [defaultdict(list) for _ in range(self.num_nodes)]
        self.metrics = defaultdict(list)
        
        # Example data for each stage
        _, _, test_data, vocab = get_data(block_size=block_size, setting=self.setting)
        dataset = SentencePairDataset(test_data, setting=self.setting)
        ntokens = len(vocab) # the size of vocabulary
        if n_samples > 0:
            n_samples = min(n_samples, len(dataset))
            if self.setting == 'random':
                indices = random.sample(range(len(dataset)), n_samples)
            elif self.setting == 'variant':
                indices = list(range(n_samples))
            dataset = Subset(
                dataset, 
                indices,
            )
            
        self.model_kwargs = {
            'nlayers': nlayers,
            'emsize': emsize,
            'nhead': nhead,
            'nhid': nhid,
            'dropout': dropout,
            'ntokens': ntokens,
        }
        
        self.dataloader = self.get_dataloader(batch_size, dataset, vocab)
        
        
    def get_dataloader(self, batch_size: int, dataset: SentencePairDataset, vocab: Dict[str, int]):
        def collate_batch(batch):
            # 'batch' is a list of tuples with (sequence, target)
            batch_data, batch_target = zip(*batch)
            combined_list = batch_data + batch_target
            # Dynamically pad the batch
            padded = pad_sequence(combined_list, batch_first=True, padding_value=vocab['<pad>'])
            padded_data = padded[:len(batch_data)]
            padded_target = padded[len(batch_data):]
            return padded_data, padded_target.view(-1)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_batch,
            shuffle=False,
        )
        return dataloader

        
    def get_preloaded_dataset(
        self,
        distributed_nodes: List[Node], 
        setting: str, 
        dataloader: DataLoader, 
        retraining_rate: float = 0.1,
    ) -> Dict[int, List[Task]]:
        print("Using preloaded data ...")
        preloaded_tasks = defaultdict(list)
        for nodeID, node in enumerate(distributed_nodes):
            if setting == 'random':
                for i, batch in enumerate(dataloader):
                    # 10% of the time, produce a task with feedback
                    if random.random() < retraining_rate:
                        require_training = True
                    else:
                        require_training = False
                        
                    task = Task(
                        task_id=i,
                        query=batch[0].cuda(node.init_device), 
                        feedback=batch[1].cuda(node.last_device), 
                        node_id=nodeID,
                        num_gpus_per_node=node.num_gpus_per_node,
                        require_training=require_training,
                    ) 
                    preloaded_tasks[node.node_id].append(task)
                    
            elif setting == 'variant':
                for i, batch in enumerate(dataloader):
                    # Odd batches are short, better utilize the bubble for retraining
                    if i % 2 == 0 and random.random() < 2 * retraining_rate:
                        require_training = True
                    else:
                        require_training = False
                    
                    task = Task(
                        task_id=i,
                        query=batch[0].cuda(node.init_device), 
                        feedback=batch[1].cuda(node.last_device), 
                        node_id=nodeID,
                        num_gpus_per_node=node.num_gpus_per_node,
                        require_training=require_training,
                    )     
                    preloaded_tasks[node.node_id].append(task)
        return preloaded_tasks


    def producer(
        self,
        taskQueue: queue.Queue, 
        dataloader: DataLoader, 
        rate_lambda: float, 
        workload: str = 'poisson',
    ):
        # Produce using the dataset
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # print(f"query shape: {batch[0].shape}, target shape: {batch[1].shape}")
            if workload == 'poisson':
                time.sleep(random.expovariate(rate_lambda))
            elif workload == 'all':
                time.sleep(0)
            else:
                raise ValueError(f"Invalid workload type: {workload}")
            # 10% of the time, produce a task with feedback
            print("Producing task {} with length {}".format(i, batch[0].size(1)))
            # Essentially, we are using preloaded data (task ID)
            taskQueue.put(i)
            
        taskQueue.put(None)  # Signal the end of the dataset
        print("Producer finished producing tasks")
        

    def globalScheduler(
        self, 
        taskQueue: queue.Queue, 
        distributed_nodes: List[Node],
    ):
        # Global scheduler
        while True:
            taskID: int = taskQueue.get() # ID
            if taskID is None:
                print("Global scheduler finished scheduling tasks")
                for node in distributed_nodes:
                    node.device_queues[0].put(None)
                break
            nodeID = random.randint(0, len(distributed_nodes) - 1)
            # Each node queue store task IDs
            distributed_nodes[nodeID].device_queues[0].put(taskID)
            print("Global scheduler scheduled task {} to node {}".format(taskID, nodeID))


    def device_inference(
        self,
        stage: PipelineStage, 
        stageID: int,
        timing_info: dict, 
        preloaded_tasks: List[Task], 
        deviceQueue: queue.Queue,
        nextdeviceQueue: queue.Queue = None,
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(),
        init_device: int = 0,
    ):
        device = stage.device
        while True:
            taskID: int = deviceQueue.get()
            if taskID is None:
                print("Stage {} finished inference".format(stage.device))
                if nextdeviceQueue is not None:
                    nextdeviceQueue.put(None)
                break
            
            task = preloaded_tasks[taskID]
            assert task.task_id == taskID
            hidden = task.hiddens[stageID]
            if hidden is None:
                print("Stage {} waiting for task {}".format(stage.device, taskID))
                continue
            
            if task.require_training:
                # This is a retraining task
                record_time(device, 'start', 'forward_loss', timing_info)
                output = stage(hidden)
                record_time(device, 'end', 'forward_loss', timing_info)
            else:
                record_time(device, 'start', 'forward', timing_info)
                with torch.no_grad():
                    output = stage(hidden)
                record_time(device, 'end', 'forward', timing_info)
            
            if nextdeviceQueue is None:
                # Backprop on the last stage
                # print("Stage {} start backward propagation for task {}".format(stage.device, taskID))
                output_flat = output.contiguous().view(-1, output.size(-1)) # (B * T, C)
                # print("output_flat shape: {}, feedback shape: {}".format(output_flat.shape, task.feedback.shape))
                loss = criterion(output_flat, task.feedback)
                self.metrics["loss"].append(loss.item())
                # print("loss: {}, start loss backward ...".format(loss))
                if task.require_training:
                    loss.backward()
                    record_time(init_device, 'end', 'backward', timing_info)
                    # print("Stage {} finish backward propagation for task {} !".format(stage.device, taskID))    
            else:
                # Need to send the output to the next stage, except for the last stage
                task.hiddens[stageID+1] = output.cuda(device+1, non_blocking=True)
                nextdeviceQueue.put(taskID)


    def node_inference(
        self,
        node: Node,
        preloaded_tasks: List[Task],
        stages: List[PipelineStage], 
        timing_info: dict,
    ):
        # We use 16 workers to simulateously get task from the queue and inference
        with ThreadPoolExecutor(max_workers=len(stages)) as executor:
            for stageID, stage in enumerate(stages):
                future = executor.submit(
                    self.device_inference, 
                    stage, 
                    stageID,
                    timing_info, 
                    preloaded_tasks,
                    node.device_queues[stageID],
                    node.device_queues[stageID+1] if stageID != len(stages) - 1 else None,
                    criterion=nn.CrossEntropyLoss(),
                    init_device=node.init_device,
                )
                
        print("Node {} finished inference".format(node.node_id))


    def run_stages_concurrently(
        self,
        preloaded_tasks: Dict[int, List[Task]],
        distributed_stages: List[List[PipelineStage]], 
        timing_infos: List[dict], 
        distributed_nodes: List[Node],
    ):
        with ThreadPoolExecutor(max_workers=len(distributed_nodes)) as executor:
            for nodeID, node in enumerate(distributed_nodes):
                future = executor.submit(
                    self.node_inference, 
                    node, 
                    preloaded_tasks[nodeID], 
                    distributed_stages[nodeID], 
                    timing_infos[nodeID],
                )
            
            
    def run(self):
        
        # Preloaded dataset
        preloaded_tasks = self.get_preloaded_dataset(
            self.distributed_nodes, 
            self.setting, 
            self.dataloader, 
            retraining_rate=self.retraining_rate,
        )

        # Instantiate stages and put them on the correct devices
        distributed_stages = [
            get_transformer_stages(
                num_stages=self.num_gpus_per_node,
                init_device=self.distributed_nodes[nodeID].init_device,
                timing_info=self.timing_infos[nodeID],
                **self.model_kwargs,
            )
            for nodeID in range(self.num_nodes)
        ]

        # Run the stages concurrently
        task_queue = queue.Queue()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future1 = executor.submit(
                self.producer,
                task_queue, 
                self.dataloader, 
                self.rate_lambda, 
                self.workload,
            )
            future2 = executor.submit(
                self.globalScheduler,
                task_queue,
                self.distributed_nodes,
            )
            future3 = executor.submit(
                self.run_stages_concurrently,  
                preloaded_tasks, 
                distributed_stages,
                self.timing_infos,
                self.distributed_nodes,
            )
            
        # Save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        stats_f = f'{self.output_dir}/metrics_{self.setting}_{self.workload}_{self.retraining_rate}.json'
        with open(stats_f, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {stats_f}:\n{self.metrics}")
            
        # Save timing info
        self.save_timing_info()


    def save_timing_info(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for nodeID, timing_info in enumerate(self.timing_infos):
            # # Remove the first start and end time for each GPU
            # gpus = list(set(int(key.split('_')[0]) for key in timing_info))
            # for gpu_id in gpus:
            #     timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start']
            #     timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end']
            stats_f = f'{self.output_dir}/timing_info_coroutine_{self.setting}_{self.workload}_{self.retraining_rate}_node{nodeID}.json'
            with open(stats_f, 'w') as f:
                json.dump(timing_info, f, indent=4)
    
        
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--setting', type=str, default='random', choices=['identical','random', 'variant'], help='workload setting')
    parser.add_argument('--nlayers', type=int, default=24)
    parser.add_argument('--emsize', type=int, default=2048)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nhid', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--retraining_rate', type=float, default=0.1)
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--workload', type=str, default='poisson', choices=['poisson', 'all'])
    parser.add_argument('--output_dir', type=str, default='prof')
    args = parser.parse_args()
    
    distributed_llm = DistributedTransformer(args)
    distributed_llm.run()
