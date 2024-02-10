import os
import sys
sys.dont_write_bytecode = True
import time
import json
import queue
import random
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    set_seed,
    get_scheduler,
)
from utils import Node, Task
from models import (
    get_stages, 
    _prepare_inputs,
)
 

class DistributedLLM:
    model_n: str = 'dummy'

    def __init__(self, args: argparse.Namespace):
        n_samples = args.n_samples
        self.setting = args.setting
        self.num_nodes = args.num_nodes
        self.batch_size = args.batch_size
        self.rate_lambda = args.rate_lambda
        self.output_dir = args.output_dir
        self.lr = args.lr
        self.workload = args.workload
        self.retraining_rate = args.retraining_rate  
        
        # Reproducibility
        set_seed(args.seed)
        self.num_gpus_per_node = torch.cuda.device_count() // self.num_nodes
        self.distributed_nodes = {
            nodeID: Node(nodeID, self.num_gpus_per_node, nodeID * self.num_gpus_per_node) 
            for nodeID in range(self.num_nodes)
        }
        self.timing_infos = {
            nodeID: defaultdict(list) 
            for nodeID in range(self.num_nodes)
        }
        self.metrics = defaultdict(list)
        
        # Load the model and tokenizer
        self.access_token = args.access_token
        self.model_name_or_path = args.model_name_or_path
        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            token=self.access_token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            token=self.access_token,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=None,
        )
        
        # Load datasets and dataloaders
        self.dataset_name_or_path = args.dataset_name_or_path
        datasets = load_dataset(self.dataset_name_or_path)
        train_dataset = datasets['train']
        test_dataset = datasets['test']
        if n_samples > 0:
            n_samples = min(n_samples, len(train_dataset), len(test_dataset))
            indices = random.sample(range(len(train_dataset)), n_samples)
            train_dataset = train_dataset.select(indices)
            indices = random.sample(range(len(test_dataset)), n_samples)
            test_dataset = test_dataset.select(indices)
        
        self.train_dataset = train_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
        ).remove_columns(datasets['train'].column_names)
        self.test_dataset = test_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
        ).remove_columns(datasets['train'].column_names)
    
        # self.train_dataloader = self.get_dataloader(dataset=self.train_dataset)
        self.test_dataloader = self.get_dataloader(dataset=self.test_dataset)
        
        # Preloaded dataset
        self.distributed_preloaded_tasks = self.get_preloaded_dataset(
            self.distributed_nodes, 
            self.test_dataloader, 
            retraining_rate=self.retraining_rate,
        )
        
        # Stages, opimizer, and scheduler
        self.distributed_stages = {
            nodeID: get_stages(
                self.config,
                token=self.access_token,
                model_name_or_path=self.model_name_or_path,
                num_stages=self.num_gpus_per_node,
                init_device=self.distributed_nodes[nodeID].init_device,
                timing_info=self.timing_infos[nodeID],
            ) for nodeID in range(self.num_nodes)
        }

        self.distributed_optimizers = {}
        for nodeID in range(self.num_nodes):
            all_parameters = []
            # Collect all parameters from stages in each node
            for stage in self.distributed_stages[nodeID]: 
                all_parameters.extend(list(stage.parameters()))
            self.distributed_optimizers[nodeID] = torch.optim.AdamW(all_parameters, lr=self.lr)
        
        self.distributed_schedulers = {
            nodeID: get_scheduler(
                "linear",
                optimizer=self.distributed_optimizers[nodeID], 
                num_warmup_steps=0, 
                num_training_steps=100,
            ) for nodeID in range(self.num_nodes)
        }
        
        
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['query'], 
            padding=False, 
            truncation=True,
        )
        labels = self.tokenizer(
            examples['reference'], 
            padding=False, 
            truncation=True,
        )
        tokenized_inputs['labels'] = labels['input_ids']
        # tokenized_inputs['labels_attention_mask'] = labels['attention_mask']
        return tokenized_inputs
        
        
    def get_dataloader(
        self, 
        batch_size: Optional[int] = None, 
        dataset: Optional[Dataset] = None,
    ) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        dataset = dataset if dataset is not None else self.train_dataset
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )

        
    def get_preloaded_dataset(
        self,
        distributed_nodes: Optional[Dict[int, Node]] = None, 
        dataloader: Optional[DataLoader] = None, 
        retraining_rate: Optional[float] = None,
    ) -> Dict[int, List[Task]]:
        print("Using preloaded data ...")
        distributed_nodes = distributed_nodes if distributed_nodes is not None else self.distributed_nodes
        dataloader = dataloader if dataloader is not None else self.test_dataloader
        retraining_rate = retraining_rate if retraining_rate is not None else self.retraining_rate
        distributed_preloaded_tasks = defaultdict(list)
        
        for nodeID, node in distributed_nodes.items():
            for i, batch in enumerate(dataloader):
                # 10% of the time, produce a task with feedback
                if random.random() < retraining_rate:
                    require_training = True
                else:
                    require_training = False  
                
                task = Task(
                    task_id=i,
                    query=_prepare_inputs(batch, device=node.init_device),
                    feedback=_prepare_inputs(batch['labels'], device=node.last_device),
                    node_id=nodeID,
                    num_gpus_per_node=node.num_gpus_per_node,
                    require_training=require_training,
                )
                distributed_preloaded_tasks[nodeID].append(task)
                
        return distributed_preloaded_tasks


    def producer(
        self,
        taskQueue: queue.Queue, 
        dataloader: Optional[DataLoader] = None, 
        rate_lambda: Optional[float] = None, 
        workload: Optional[str] = None,
    ):
        dataloader = dataloader if dataloader is not None else self.test_dataloader
        rate_lambda = rate_lambda if rate_lambda is not None else self.rate_lambda
        workload = workload if workload is not None else self.workload
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
            # print("Producing task {} with input shape {}".format(i, batch['input_ids'].shape))
            # Essentially, we are using preloaded data (task ID)
            taskQueue.put(i)
            
        taskQueue.put(None)  # Signal the end of the dataset
        print("Producer finished producing tasks")
        

    def globalScheduler(
        self, 
        taskQueue: queue.Queue, 
        distributed_nodes: Optional[Dict[int, Node]] = None,
    ):
        distributed_nodes = distributed_nodes if distributed_nodes is not None else self.distributed_nodes
        # Global scheduler
        while True:
            taskID: int = taskQueue.get() # ID
            if taskID is None:
                print("Global scheduler finished scheduling tasks")
                for node in distributed_nodes.values():
                    node.device_queues[0].put(None)
                break
            nodeID = random.choice(list(distributed_nodes.keys()))
            # Each node queue store task IDs
            distributed_nodes[nodeID].device_queues[0].put(taskID)
            # print("Global scheduler scheduled task {} to node {}".format(taskID, nodeID))
            
            
    def forward(
        self, 
        task: Task,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        stage: Union[torch.nn.Module],
        device: int, 
        timing_info: Dict[str, List[float]],
    ):
        raise NotImplementedError("forward method must be implemented")
    

    def device_inference(
        self,
        stage: torch.nn.Module, 
        stageID: int,
        nodeID: int,
        timing_info: Dict[str, List[float]],
        preloaded_tasks: List[Task], 
        deviceQueue: queue.Queue,
        nextdeviceQueue: Optional[queue.Queue] = None,
        init_device: Optional[int] = None,
    ):
        raise NotImplementedError("device_inference method must be implemented")             


    def node_inference(
        self,
        nodeID: int,
        node: Node,
        preloaded_tasks: List[Task],
        stages: List[torch.nn.Module], 
        timing_info: Dict[str, List[List[float]]],
    ):
        # We use 16 workers to simulateously get task from the queue and inference
        with ThreadPoolExecutor(max_workers=len(stages)) as executor:
            for stageID, stage in enumerate(stages):
                future = executor.submit(
                    self.device_inference, 
                    stage=stage, 
                    stageID=stageID,
                    nodeID=nodeID,
                    timing_info=timing_info, 
                    preloaded_tasks=preloaded_tasks,
                    deviceQueue=node.device_queues[stageID],
                    nextdeviceQueue=node.device_queues[stageID+1] if stageID != len(stages) - 1 else None,
                    init_device=node.init_device,
                )
                
        print("Node {} finished inference".format(node.node_id))


    def run_stages_concurrently(
        self,
        distributed_preloaded_tasks: Optional[Dict[int, List[Task]]] = None,
        distributed_stages: Optional[Dict[int, List[torch.nn.Module]]] = None, 
        timing_infos: Optional[Dict[int, dict]] = None, 
        distributed_nodes: Optional[Dict[int, Node]] = None,
    ):
        distributed_preloaded_tasks = distributed_preloaded_tasks if distributed_preloaded_tasks is not None else self.distributed_preloaded_tasks
        distributed_stages = distributed_stages if distributed_stages is not None else self.distributed_stages
        timing_infos = timing_infos if timing_infos is not None else self.timing_infos
        distributed_nodes = distributed_nodes if distributed_nodes is not None else self.distributed_nodes
        
        with ThreadPoolExecutor(max_workers=len(distributed_nodes)) as executor:
            for nodeID, node in distributed_nodes.items():
                future = executor.submit(
                    self.node_inference, 
                    nodeID,
                    node, 
                    distributed_preloaded_tasks[nodeID], 
                    distributed_stages[nodeID], 
                    timing_infos[nodeID],
                )
            
            
    def run(self):
        # Run the stages concurrently
        task_queue = queue.Queue()
        with ThreadPoolExecutor(max_workers=3) as executor:
            future1 = executor.submit(
                self.producer,
                task_queue, 
                dataloader=self.test_dataloader,
            )
            future2 = executor.submit(
                self.globalScheduler,
                task_queue,
            )
            future3 = executor.submit(self.run_stages_concurrently)
            
        # Save timing info
        self.save_timing_info()
        
        # Calculate metrics
        self.calculate_metrics()
        
        
    def save_timing_info(
        self, 
        timing_infos: Optional[Dict[int, dict]] = None,
    ):
        timing_infos = timing_infos if timing_infos is not None else self.timing_infos
        os.makedirs(self.output_dir, exist_ok=True)
        for nodeID, timing_info in timing_infos.items():
            # # Remove the first start and end time for each GPU
            # gpus = list(set(int(key.split('_')[0]) for key in timing_info))
            # for gpu_id in gpus:
            #     timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start']
            #     timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end']
            stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{self.setting}_{self.workload}_{self.retraining_rate}_node{nodeID}.json'
            with open(stats_f, 'w') as f:
                json.dump(timing_info, f, indent=4)
        
        
    def calculate_metrics(
        self, 
        metrics: Optional[Dict[str, Union[float, int]]] = None,
    ):
        metrics = metrics if metrics is not None else self.metrics
        # Calculate metrics
        global_min_time, global_max_time = float('inf'), float('-inf')
        total_idles = []
        total_latencies = []
        for nodeID, node in self.distributed_nodes.items():
            timing_info = self.timing_infos[nodeID]
            
            for gpu_id in range(self.num_gpus_per_node):
                min_t, max_t = float('inf'), float('-inf')
                gpu_idx = node.init_device + gpu_id
                starts = timing_info.get(f"{gpu_idx}_start", [])
                ends = timing_info.get(f"{gpu_idx}_end", [])
                if len(starts) == 1:
                    idles = [0]
                else:
                    idles = [start - end for (start, start_label), (end, end_label) in zip(starts[1:], ends[:-1]) if (start_label == end_label and start > end)]
                total_idles.extend(idles)
                
                tasks = list(zip(starts, ends))
                for i, ((start, start_label), (end, _)) in enumerate(tasks):
                    metrics[start_label].append(end - start)
                    min_t = min(min_t, start)
                    max_t = max(max_t, end)
                total_latencies.append(max_t - min_t)
                global_min_time = min(global_min_time, min_t)
                global_max_time = max(global_max_time, max_t)
                    
        num_tasks = len(self.distributed_preloaded_tasks[0])
        bubble_rate = sum(total_idles) / sum(total_latencies) if sum(total_latencies) > 0 else 0
        for key, value in metrics.items():
            metrics[key] = sum(value) / len(value)
        
        metrics['num_tasks'] = num_tasks
        metrics['bubble_rate'] = bubble_rate 
        metrics['idleness'] = sum(total_idles) / len(total_idles)
        metrics['response_time'] = sum(total_latencies) * 2 / (num_tasks * len(total_latencies))
        metrics['end2end_latency'] = global_max_time - global_min_time
        metrics['throughput'] = num_tasks / (global_max_time - global_min_time)
            
        # Save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        stats_f = f'{self.output_dir}/metrics_{self.model_n}_{self.setting}_{self.workload}_{self.retraining_rate}.json'
        with open(stats_f, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {stats_f}:\n{metrics}")

    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, help='model name or path')
    parser.add_argument('--access_token', type=str, default=None, help='access token')
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--setting', type=str, default='active', choices=['active','interval', 'one_node'], help='training setting')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--workload', type=str, default='poisson', choices=['poisson', 'all'], help='workload arrival pattern')
    parser.add_argument('--output_dir', type=str, default='prof')
    args = parser.parse_args()
    
    distributed_llm = DistributedLLM(args)
    distributed_llm.run()
