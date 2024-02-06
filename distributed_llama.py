import os
import sys
sys.dont_write_bytecode = True
import time
import json
import queue
import random
import argparse
import torch
import pdb
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union
from collections import defaultdict
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoTokenizer, default_data_collator
from utils import record_time
from models import (
    get_stages, 
    LlamaStartingStage,
    LlamaIntermediateStage,
    LlamaEndingStage, 
    _prepare_inputs,
    compute_nll_loss,
    CustomizedOut,
    CausalLMOutputWithPast,
)



class Node:
    def __init__(
        self, 
        node_id: int, 
        num_gpus_per_node: int, 
        init_device: int = 0,
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
        query: Dict[str, Union[torch.Tensor, Any]], 
        feedback: Dict[str, Union[torch.Tensor, Any]], 
        node_id: Optional[int] = None, 
        num_gpus_per_node: Optional[int] = None,
        require_training: Optional[bool] = None,
    ):
        self.task_id = task_id
        self.feedback = feedback
        num_gpus_per_node = num_gpus_per_node if num_gpus_per_node is not None else 1
        self.hiddens = [query] + [None for _ in range(num_gpus_per_node - 1)]
        self.node_id = node_id if node_id is not None else 0
        self.require_training = False if require_training is None else require_training
        

class DistributedLLM:
    
    def __init__(self, args: argparse.Namespace):
        n_samples = args.n_samples
        self.setting = args.setting
        self.num_nodes = args.num_nodes
        self.batch_size = args.batch_size
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
        
        # Load the dataset
        datasets = load_dataset('data/Anthropic')
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
        )
        self.test_dataset = test_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
        )
        # Batch: input_ids, attention_mask, labels, labels_attention_mask
        self.train_dataloader = self.get_dataloader(dataset=self.train_dataset)
        self.test_dataloader = self.get_dataloader(dataset=self.test_dataset)
    
        
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['query'], 
            padding=True, 
            truncation=True,
        )
        labels = self.tokenizer(
            examples['reference'], 
            padding=True, 
            truncation=True,
        )
        tokenized_inputs['labels'] = labels['input_ids']
        tokenized_inputs['labels_attention_mask'] = labels['attention_mask']
        return tokenized_inputs
        
        
    def get_dataloader(
        self, 
        batch_size: Optional[int] = None, 
        dataset: Optional[Dataset] = None,
    ):
        batch_size = batch_size if batch_size is not None else self.batch_size
        dataset = dataset if dataset is not None else self.train_dataset
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )

        
    def get_preloaded_dataset(
        self,
        distributed_nodes: List[Node], 
        dataloader: DataLoader, 
        retraining_rate: float = 0.1,
    ) -> Dict[int, List[Task]]:
        print("Using preloaded data ...")
        preloaded_tasks = defaultdict(list)
        for nodeID, node in enumerate(distributed_nodes):
            for i, batch in enumerate(dataloader):
                labels = {
                    'input_ids': batch.pop('labels'),
                    'attention_mask': batch.pop('labels_attention_mask'),
                }
                # 10% of the time, produce a task with feedback
                if random.random() < retraining_rate:
                    require_training = True
                else:
                    require_training = False  
                
                task = Task(
                    task_id=i,
                    # query=batch[0].cuda(node.init_device), 
                    query=_prepare_inputs(batch, device=node.init_device),
                    # feedback=batch[1].cuda(node.last_device), 
                    feedback=_prepare_inputs(labels, device=node.last_device),
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
        stage: LlamaEndingStage, 
        stageID: int,
        timing_info: dict, 
        preloaded_tasks: List[Task], 
        deviceQueue: queue.Queue,
        nextdeviceQueue: queue.Queue = None,
        init_device: int = 0,
    ):
        device = stage.device
        while True:
            taskID: int = deviceQueue.get()
            if taskID is None:
                print("Stage {} finished inference".format(device))
                if nextdeviceQueue is not None:
                    nextdeviceQueue.put(None)
                break
            
            task = preloaded_tasks[taskID]
            assert task.task_id == taskID
            inputs = task.hiddens[stageID]
            if inputs is None:
                print("Stage {} waiting for task {}".format(device, taskID))
                continue
            
            if nextdeviceQueue is not None: # intermediate stage
                if task.require_training: # this is a retraining task
                    record_time(device, 'start', 'forward_loss', timing_info)
                    tuple_outputs = stage(**inputs)
                    record_time(device, 'end', 'forward_loss', timing_info)
                else:
                    record_time(device, 'start', 'forward', timing_info)
                    with torch.no_grad():
                        tuple_outputs = stage(**inputs)
                    record_time(device, 'end', 'forward', timing_info)
                
                outputs = CustomizedOut(
                    hidden_states=tuple_outputs[0],
                    past_key_values=tuple_outputs[1],
                    all_hidden_states=tuple_outputs[2],
                    all_self_attns=tuple_outputs[3],
                    position_ids=tuple_outputs[4],
                    attention_mask=tuple_outputs[5],
                )
                # Need to send the output to the next stage, except for the last stage
                task.hiddens[stageID+1] = _prepare_inputs(outputs, device+1)
                nextdeviceQueue.put(taskID)
                
            else: # ending stage
                
                if task.require_training: # this is a retraining task
                    record_time(device, 'start', 'forward_loss', timing_info)
                    nll_loss = compute_nll_loss(stage, inputs, task.feedback)
                    record_time(device, 'end', 'forward_loss', timing_info)
                else:
                    record_time(device, 'start', 'forward', timing_info)
                    with torch.no_grad():
                        nll_loss = compute_nll_loss(stage, inputs, task.feedback)
                    record_time(device, 'end', 'forward', timing_info)
                    
                print("[loss={}] stage {} finished inference for task {}".format(
                    nll_loss, device, taskID
                ))
                
                if self.setting == 'active':
                    # Backprop on the last stage
                    # print("Stage {} start backward propagation for task {}".format(device, taskID))
                    # print("output_flat shape: {}, feedback shape: {}".format(output_flat.shape, task.feedback.shape))
                    # print("loss: {}, start loss backward ...".format(nll_loss))
                    nll_loss.backward()
                    record_time(init_device, 'end', 'backward', timing_info)
                    # print("Stage {} finish backward propagation for task {} !".format(device, taskID))
                else:
                    task.hiddens.append(nll_loss)
                    deviceQueue.put(taskID) # put it back to the queue


    def node_inference(
        self,
        node: Node,
        preloaded_tasks: List[Task],
        stages: List[LlamaEndingStage], 
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
                    deviceQueue=node.device_queues[stageID],
                    nextdeviceQueue=node.device_queues[stageID+1] if stageID != len(stages) - 1 else None,
                    init_device=node.init_device,
                )
                
        print("Node {} finished inference".format(node.node_id))


    def run_stages_concurrently(
        self,
        preloaded_tasks: Dict[int, List[Task]],
        distributed_stages: List[List[LlamaEndingStage]], 
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
            self.test_dataloader, 
            retraining_rate=self.retraining_rate,
        )
        
        distributed_stages = [
            get_stages(
                self.config,
                token=self.access_token,
                model_name_or_path=self.model_name_or_path,
                num_stages=self.num_gpus_per_node,
                init_device=self.distributed_nodes[nodeID].init_device,
                timing_info=self.timing_infos[nodeID],
            ) for nodeID in range(self.num_nodes)
        ]
        # # get a batch
        # print("Node [0] preloaded task [0]: ", vars(preloaded_tasks[0][0]))
        # pdb.set_trace()

        # Run the stages concurrently
        task_queue = queue.Queue()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future1 = executor.submit(
                self.producer,
                task_queue, 
                self.test_dataloader, 
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
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='model name or path')
    parser.add_argument('--access_token', type=str, default=None, help='access token')
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--setting', type=str, default='active', choices=['active','interval', 'one_node'], help='training setting')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1)
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--workload', type=str, default='poisson', choices=['poisson', 'all'], help='workload arrival pattern')
    parser.add_argument('--output_dir', type=str, default='prof')
    args = parser.parse_args()
    
    distributed_llm = DistributedLLM(args)
    distributed_llm.run()
