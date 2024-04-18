import os
import sys
sys.dont_write_bytecode = True
import pdb
import time
import scipy
import json
import queue
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm
import GPUtil
from GPUtil import GPU
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    set_seed,
    get_scheduler,
)
from utils import Node, Task, record_time
from models import (
    get_stages, 
    _prepare_inputs,
)


# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 

class DistributedLLM:

    def __init__(self, args: argparse.Namespace):
        self.n_samples = args.n_samples
        self.setting = args.setting
        self.priority = args.priority
        self.num_nodes = args.num_nodes
        self.batch_size = args.batch_size
        self.train_lambda = args.train_lambda
        self.test_lambda = args.test_lambda
        self.output_dir = args.output_dir
        self.task_assignment = args.task_assignment
        self.dataset_name_or_path = args.dataset_name_or_path
        self.lr = args.lr
        self.workload = args.workload
        self.retraining_rate = args.retraining_rate  
        self.model_n = args.model_name
        self.save_length = args.save_length
        self.length_distribution = args.length_distribution
        self.length_heterogeneity = args.length_heterogeneity
        self.active_selection = args.active_selection
        self.eta = None # forward coefficient
        self.BT = None # backward latency
        self.alpha = args.alpha
        self.beta = args.beta
        self.epsilon = args.epsilon
        
        if self.setting == 'isolated':
            self.isolated_split = args.isolated_split if args.isolated_split is not None else self.retraining_rate
            setting = f"isolated-split{self.isolated_split}"
            
            num_train_nodes = max(1, round(self.num_nodes * self.isolated_split))
            num_test_nodes = max(1, self.num_nodes - num_train_nodes)
            self._test_nodes = list(range(num_test_nodes))
            self._train_nodes = list(range(num_test_nodes, self.num_nodes))
            print(f"** ISOLATED SYSTEM: Test nodes: {self._test_nodes}, Train nodes: {self._train_nodes} **")
        else:
            setting = self.setting
            self._train_nodes = list(range(self.num_nodes))
            self._test_nodes = list(range(self.num_nodes))
        
        # task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta})" if self.task_assignment == 'workload' else self.task_assignment
        if self.priority is not None:
            self.ckpt_path = f'{self.output_dir}/stages-{self.task_assignment}_{self.model_n}_{setting}-{self.priority}_{self.workload}_{self.retraining_rate}'
        else:
            self.ckpt_path = f'{self.output_dir}/stages-{self.task_assignment}_{self.model_n}_{setting}_{self.workload}_{self.retraining_rate}'

        self.task_arrival = defaultdict(dict)
        self.task_trace = defaultdict(lambda: defaultdict(dict))
        self.train_trace = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.user_task_record = defaultdict(dict)
        self.all_trace = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.node_allocation = defaultdict(list)
        self.node_accumulated_bubble = defaultdict(int)
        self.forward_start = torch.cuda.Event(enable_timing=True)
        self.forward_end = torch.cuda.Event(enable_timing=True)
        self.backward_start = torch.cuda.Event(enable_timing=True)
        self.backward_end = torch.cuda.Event(enable_timing=True)
        
        # Reproducibility
        set_seed(args.seed)
        self.num_gpus_per_node = torch.cuda.device_count() // self.num_nodes
        self.distributed_nodes = {
            nodeID: Node(
                nodeID, 
                self.num_gpus_per_node, 
                init_device=nodeID * self.num_gpus_per_node,
            ) for nodeID in range(self.num_nodes)
        }
        self.memory_threshold = args.memory_threshold
        self.device_total_memory = torch.cuda.get_device_properties(0).total_memory
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
        datasets = load_dataset(self.dataset_name_or_path)
        # train_dataset = datasets['train']
        test_dataset = datasets['test']
        
        # self.train_dataset = train_dataset.map(
        #     self._tokenize_and_align_labels,
        #     batched=True,
        # ).remove_columns(datasets['train'].column_names)
        test_dataset = test_dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
        ).remove_columns(datasets['train'].column_names)
        
        # Do sampling according to the length distribution
        input_lengths = [len(x) for x in test_dataset['input_ids']]
        self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length = np.mean(input_lengths), np.std(input_lengths), np.median(input_lengths), min(input_lengths), max(input_lengths)
        print(" ** Original data length distribution: mean={}, std={}, medium={}, min={}, max={} **".format(
            self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length))
        if self.n_samples > 0:
            n_samples = min(self.n_samples, len(input_lengths))
            if self.length_heterogeneity is None:
                indices = random.sample(range(len(input_lengths)), n_samples)
                test_dataset = test_dataset.select(indices)
            else:
                indices = self._sample_subset_indices(input_lengths, n_samples, self.mean_length, self.length_heterogeneity)
                test_dataset = test_dataset.select(indices)
  
            subset_lengths = [len(x) for x in test_dataset['input_ids']]
            self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length = np.mean(subset_lengths), np.std(subset_lengths), np.median(subset_lengths), min(subset_lengths), max(subset_lengths)
            print(f" ** Sampled {len(subset_lengths)} data points: mean={self.mean_length}, std={self.std_length}, medium={self.medium_length}, min={self.min_length}, max={self.max_length} **")
        
        self.test_dataset = test_dataset
    
        # self.train_dataloader = self.get_dataloader(dataset=self.train_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )
        
        # Preloaded dataset
        self.distributed_preloaded_tasks, self.total_tasks, self.retraining_tasks = self.get_preloaded_dataset(
            self.distributed_nodes, 
            self.test_dataloader, 
            retraining_rate=self.retraining_rate,
        )
        self.saving_steps = max(min(100, self.retraining_tasks // 2), 1)
        print(f" ** Total tasks: {self.total_tasks}, retraining tasks: {self.retraining_tasks}, saving steps: {self.saving_steps} ** ")
        self._training_step = 0
        self._trained_task_lengths = []
        
        # Save task length distribution for further analysis
        if self.save_length:
            length_dict = {taskID: task.query['input_ids'].shape[1] for taskID, task in enumerate(self.distributed_preloaded_tasks[0])}
            with open(f"{self.output_dir}/task_length_{self.model_n}_{setting}_{self.workload}_{self.retraining_rate}.json", 'w') as f:
                json.dump(length_dict, f, indent=4)
        
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
        
        
    def _tokenize_and_align_labels(self, examples):
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
    
    
    def _sample_subset_indices(self, input_lengths: List[int], K: int, mu: float, std: float) -> List[int]:
        # Create an empty list to store the selected numbers
        selected_ids = set()
        lengths_dict = {} # {length: [idx1, idx2, ...]}
        for idx, length in enumerate(input_lengths):
            if length not in lengths_dict:
                lengths_dict[length] = [idx]
            else:
                lengths_dict[length].append(idx)

        # We draw K samples from the normal distribution
        for _ in range(K):
            sample = np.random.normal(mu, std)
            if sample in lengths_dict:
                selected_ids.add(lengths_dict[sample][0])
                lengths_dict[sample].pop(0) # pop the selected index
                if len(lengths_dict[sample]) == 0:
                    del lengths_dict[sample]
            else:
                # Find the number in 'numbers' that is closest to the sampled number
                closest_number = min(list(lengths_dict.keys()), key=lambda x: abs(x - sample))
                selected_ids.add(lengths_dict[closest_number][0])
                lengths_dict[closest_number].pop(0)
                if len(lengths_dict[closest_number]) == 0:
                    del lengths_dict[closest_number]
            
        return selected_ids

        
    def get_preloaded_dataset(
        self,
        distributed_nodes: Optional[Dict[int, Node]] = None, 
        dataloader: Optional[DataLoader] = None, 
        retraining_rate: Optional[float] = None,
    ) -> Tuple[Dict[int, List[Task]], int, int]:
        
        print("Using preloaded data ...")
        distributed_nodes = distributed_nodes if distributed_nodes is not None else self.distributed_nodes
        dataloader = dataloader if dataloader is not None else self.test_dataloader
        retraining_rate = retraining_rate if retraining_rate is not None else self.retraining_rate
        distributed_preloaded_tasks = defaultdict(list)
        total_tasks, retraining_tasks = 0, 0
        
        selected_data = []
        for i, batch in enumerate(dataloader):
            seq_length = batch['input_ids'].shape[1]
            selected_data.append((seq_length, batch))
            
        # Define the order of arrival input sequence length
        if self.length_distribution == 'ascending':
            selected_data.sort(key=lambda x: x[0])
        elif self.length_distribution == 'descending':
            selected_data.sort(key=lambda x: x[0], reverse=True)
        elif self.length_distribution == 'bursty': # one short one long, ...
            selected_data.sort(key=lambda x: x[0])
            mid_index = len(selected_data) // 2
            short_data, long_data = selected_data[:mid_index], selected_data[mid_index:]
            # Rearrange sentences in groups of bptt
            tmp = []
            bptt = 1
            for i in range(0, max(len(short_data), len(long_data)), 1):
                tmp.extend(short_data[i:i+bptt])
                tmp.extend(long_data[i:i+bptt])
            selected_data = tmp
        elif self.length_distribution == 'random':
            pass
        else:
            raise ValueError(f"Invalid length distribution: {self.length_distribution}")
            
        # Create preloaded tasks with each one on a specific CUDA device
        for i, (_, batch) in enumerate(selected_data):
            # 10% of the time, produce a task with feedback
            require_training = random.random() < retraining_rate
            total_tasks += 1
            retraining_tasks += 1 if require_training else 0
                
            for nodeID, node in distributed_nodes.items():
                task = Task(
                    task_id=i,
                    # rate_lambda=lambda_values[i] if self.rate_lambda is None else self.rate_lambda,
                    rate_lambda=self.train_lambda if require_training else self.test_lambda,
                    query=_prepare_inputs(batch, device=node.init_device),
                    feedback=_prepare_inputs(batch['labels'], device=node.last_device),
                    node_id=nodeID,
                    num_gpus_per_node=node.num_gpus_per_node,
                    require_training=require_training,
                )
                distributed_preloaded_tasks[nodeID].append(task)
        
        return distributed_preloaded_tasks, total_tasks, retraining_tasks


    def producer(
        self,
        taskQueue: queue.Queue, 
    ):
        # Produce using the dataset
        for taskID, task in enumerate(self.distributed_preloaded_tasks[0]):
            if self.workload == 'poisson':
                time.sleep(random.expovariate(task.rate_lambda))
            elif self.workload == 'all':
                time.sleep(0)
            else:
                raise ValueError(f"Invalid workload type: {self.workload}")
            # 10% of the time, produce a task with feedback
            # print("Producing task {} with input length {}".format(taskID, task.query['input_ids'].shape[1]))
            # Essentially, we are using preloaded data (task ID)
            taskQueue.put(taskID)
            # We record and calculate the response time for each user task (no retraining)
            release = time.time()
            if not task.require_training:
                self.user_task_record[taskID]['release'] = release
            self.task_arrival[taskID]['release'] = release
            
        taskQueue.put(None)  # Signal the end of the dataset
        print("Producer finished producing tasks")
        
        
    def _compute_rewards(self, nodeID: int, taskID: int):
        """
        Predict the execution traces for the incoming task and compute the reward (bubble rate & response time).
        """
        length = self.distributed_preloaded_tasks[nodeID][taskID].query['input_ids'].shape[1]
        require_training = self.distributed_preloaded_tasks[nodeID][taskID].require_training
        fb = self.task_arrival[taskID]['release']
        br, bi, init_id, finished_ids = 0, 0, 0, []
        previous_taskIDs = list(self.all_trace[nodeID].keys())
        previous_train_taskIDs = list(self.train_trace[nodeID].keys())
        eta = self.eta if self.eta is not None else 0.002
        BT = self.BT if self.BT is not None else 0.005
            
        # Get/Estimate previous forward execution traces
        if previous_taskIDs:
            last_taskID = previous_taskIDs[-1]
            arrival_gap = self.task_arrival[taskID]['release'] - self.task_arrival[last_taskID]['release']
        else:
            last_taskID = None
            arrival_gap = 0
            
        for stageID in range(self.num_gpus_per_node):
            pfe = self.all_trace[nodeID][last_taskID][stageID]['fe'] if last_taskID is not None else fb
            fb = max(pfe, fb)
            fe = fb + eta * length ** 2
            i = init_id
            
            for i in range(init_id, len(previous_train_taskIDs)):
                ptID = previous_train_taskIDs[i]
                if fe <= self.train_trace[nodeID][ptID][stageID]['bb']:
                    br += eta * length ** 2 # update BR
                    break
                fb = max(fb, self.train_trace[nodeID][ptID][stageID]['be']) # forward starts at least at the end of the backward
                fe = fb + eta * length ** 2
                if stageID == 0: # this training task is finished
                    finished_ids.append(ptID)
                    
            # Write the forward and backward execution traces
            self.task_trace[nodeID][stageID]['fb'] = fb
            self.task_trace[nodeID][stageID]['fe'] = fe
            bi += max(fb - pfe - (i - init_id) * BT, 0) # update BI
            init_id = i + 1 # update the initial training check index for the next stage
        
        if require_training:
            bb = fe # backward starts at the end of the forward
            for stageID in range(self.num_gpus_per_node-1, -1, -1):
                self.task_trace[nodeID][stageID]['bb'] = bb
                self.task_trace[nodeID][stageID]['be'] = bb + BT
                # bi += max(bb - self.task_trace[nodeID][stageID]['fe'], 0) # update BI
                bb += BT
        
        # Clear the finished training tasks
        for ptID in finished_ids:
            self.train_trace[nodeID].pop(ptID)
            self.all_trace[nodeID].pop(ptID)
                    
        # Compute reward (bubble profit and response time)
        response = fe - self.task_arrival[taskID]['release']
        LH = self.calculate_node_LH(nodeID, length) if self.beta != 0 else 0
        # if arrival_gap is not None:
        #     reward = (br - bi / (1000 * arrival_gap)) / (self.alpha * response + self.beta * LH + self.epsilon)
        # else:
        #     reward = (br - bi) / (self.alpha * response + self.beta * LH + self.epsilon)
        # Normalize BI by the arrival gap
        # original_bi = bi
        # bi = max(1/self.test_lambda - eta * length ** 2 - self.epsilon, 0)
        # if last_taskID is None: # the first task
        #     fbi = 0
        # else:
            # fbi = max(bi/self.num_gpus_per_node - arrival_gap - self.epsilon, 0)
        fbi = max(self.epsilon, bi/self.num_gpus_per_node - arrival_gap)
        reward = (br - fbi) / (self.alpha * response + self.beta * LH + self.epsilon)
        # reward = br / (self.alpha * response + self.beta * LH + self.epsilon)
        print(f"[Node {nodeID} | Task {taskID} | reward {reward:.2f}]: epsilon {self.epsilon} - BI {bi:.2f} (BI/N - |a - a0|: {bi/self.num_gpus_per_node - arrival_gap:.2f}) - BR {br:.2f} - Response {response:.2f} - LH {LH:.2f}")
        # reward = br / (self.alpha * response + self.beta * LH + self.epsilon)
        # if len(self.node_allocation[nodeID]) > 0: # this node is not empty
        #     reward = (self.node_accumulated_bubble[nodeID] + bi) / (len(self.node_allocation[nodeID]) + 1)
        # else:
        #     reward = 10000 # we prefer to assign a task to the empty node
        # reward = br - bi # only consider bubble rate
        # blank = self.node_accumulated_bubble[nodeID] + bi
        # return reward, br, bi
        return reward
            
        
        
    def _assign_node(self, node_list: List[int], taskID: int):
        if len(node_list) == 1:
            return node_list[0]
        if self.task_assignment == 'random': # Random assignment
            return random.choice(node_list)
        elif self.task_assignment == 'rr': # Round-robin
            return node_list[taskID % len(node_list)]
        elif self.task_assignment == 'workload':
            # # Choose the node with the least workload (number of tasks in the first device queue)
            # return min(node_list, key=lambda nodeID: self.distributed_nodes[nodeID].device_queues[0].qsize())
            reward_list, bi_list, br_list = [], [], []
            for nodeID in node_list:
                try:
                    reward = self._compute_rewards(nodeID, taskID)
                except Exception as e:
                    reward = random.random()
                    logging.error(f"[node {nodeID} | task {taskID}] Bubble calculation error occurred: {e}")
                # reward = self._compute_rewards(nodeID, taskID)
                reward_list.append(reward)
                # br_list.append(br)
                # bi_list.append(bi)
            
            # Greedy: assign task to the node with the highest reward (lowest average utilization)
            best_index = np.argmax(reward_list) 
            # make it more smart, if some reward values are the same, randomly choose one
            # best_index = np.random.choice(np.where(reward_list == np.max(reward_list))[0])
            # # Instead of greedy, let's use sampling with probability proportional to the reward
            # reward_list = F.softmax(torch.FloatTensor(reward_list), dim=0).numpy()  
            # best_index = np.random.choice(range(len(reward_list)), p=reward_list)
            nodeID = node_list[best_index]
            # self.node_accumulated_bubble[nodeID] += bi_list[best_index] - br_list[best_index]
            return nodeID 
        
        elif self.task_assignment == 'util':
            # Choose the node with the least average utilization across all its GPUs
            gpus: List[GPU] = GPUtil.getGPUs()
            return min(node_list, key=lambda nodeID: np.mean([gpus[device].memoryUtil for device in self.distributed_nodes[nodeID].device_ids]))
            # return min(node_list, key=lambda nodeID: np.mean([torch.cuda.memory_allocated(device) for device in self.distributed_nodes[nodeID].device_ids]))
            
        else:
            raise ValueError(f"Invalid task assignment method: {self.task_assignment}")
        

    def globalScheduler(
        self, 
        taskQueue: queue.Queue, 
    ):
        # Global scheduler
        while True:
            taskID: int = taskQueue.get() # ID
            if taskID is None:
                print("Global scheduler finished scheduling tasks")
                for node in self.distributed_nodes.values():
                    # node.device_queues[0].put(None)
                    node.device_queues[0].put((float('inf'), float('inf'))) # for priority_queue, use a large number to signal the end
                break
            
            # Task assignment
            assign_start = time.time()
            require_training = self.distributed_preloaded_tasks[0][taskID].require_training
            try:
                if self.setting != 'isolated':
                    nodeID = self._assign_node(node_list=list(range(self.num_nodes)), taskID=taskID)
                else:
                    if require_training:
                        nodeID = self._assign_node(node_list=self._train_nodes, taskID=taskID) # assign to one of the training nodes
                    else:
                        nodeID = self._assign_node(node_list=self._test_nodes, taskID=taskID) # assign to one of the test nodes
            except Exception as e:
                logging.error(f"[Task {taskID}] Node assignment error occurred: {e}")
                nodeID = random.choice(list(range(self.num_nodes)))
                    
            # Update train trace and all trace with the task trace
            if self.task_assignment == 'workload':
                self.all_trace[nodeID][taskID] = self.task_trace[nodeID].copy()
                if require_training:
                    self.train_trace[nodeID][taskID] = self.task_trace[nodeID].copy()
                        
            self.metrics['node_assignment'].append(time.time() - assign_start)

            # Each node queue store task IDs
            if self.setting == 'interval':
                seq_length = self.distributed_preloaded_tasks[0][taskID].query['input_ids'].shape[1]
                if self.priority is None or self.priority == 'LLF':
                    priority = seq_length
                elif self.priority == 'MLF':
                    priority = -seq_length
                else:
                    raise ValueError(f"Invalid priority type: {self.priority}")
                self.distributed_nodes[nodeID].device_queues[0].put((priority, taskID))
            else:
                self.distributed_nodes[nodeID].device_queues[0].put((taskID, taskID))
            
            # Record the node allocation 
            self.node_allocation[nodeID].append({'taskID': taskID, 'type': 'training' if require_training else 'test', 'length': self.distributed_preloaded_tasks[nodeID][taskID].query['input_ids'].shape[1]})
            # print("Global scheduler scheduled task {} (requre_training={}) to node {}".format(taskID, self.distributed_preloaded_tasks[0][taskID].require_training, nodeID))
    
    
    def _check_device_availability(self, device: int, threshold: float = 0.8):
        """
        Check if the device has enough available memory.
        Args:
        - device: The device to check.
        - threshold: The maximum allowed memory utilization ratio.
        Returns:
        - is_available: Boolean indicating if the device is available.
        """
        # Get device memory status
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = self.device_total_memory - allocated_memory
        # Calculate the available memory ratio
        available_ratio = available_memory / self.device_total_memory
        # Check if the available memory ratio is above the threshold
        return available_ratio > (1 - threshold)
    
    
    def _wait_for_device_availability(self, device: int, check_interval: float = 0.1, threshold: float = 0.8):
        """
        Wait until the device is available based on memory usage.
        Args:
        - device: The device to wait for.
        - check_interval: How often to check the device status (in seconds).
        - threshold: The maximum allowed memory utilization ratio.
        """
        wait_count = 0
        while not self._check_device_availability(device, threshold):
            # print(f"Waiting for device {device} to become available...")
            time.sleep(check_interval)
            wait_count += 1
            if wait_count > 100:
                print(f"Device {device} is not available after waiting for {wait_count * check_interval} seconds")
                break

            
    def forward(
        self, 
        task: Task,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        stageID: int,
        nodeID: int,
        device: int, 
        timing_info: Dict[str, List[float]],
    ) -> Tuple[torch.Tensor, ...]:
        
        # if stageID == 0:
        #     self.task_trace[nodeID].clear() # clear the current task trace
        try:
            self._wait_for_device_availability(device, threshold=self.memory_threshold)
            if self.eta is None:
                self.forward_start.record()
            if task.require_training: # this is a retraining task
                fb = record_time(device, 'start', 'forward_grad', task.task_id, timing_info)
                self.task_trace[nodeID][stageID]['fb'] = fb
                if task.task_id in self.train_trace[nodeID]: # this training task has been completely recorded
                    self.train_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task
                    self.all_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task

                tuple_outputs = self.distributed_stages[nodeID][stageID](**inputs, labels=task.feedback)
                # torch.cuda.synchronize() # synchronize the device
                fe = record_time(device, 'end', 'forward_grad', task.task_id, timing_info)
                
                self.task_trace[nodeID][stageID]['fe'] = fe
                if task.task_id in self.train_trace[nodeID]: # this training task has been completely recorded
                    self.train_trace[nodeID][task.task_id][stageID]['fe'] = fe # update the already recorded task
                    self.all_trace[nodeID][task.task_id][stageID]['fe'] = fe # update the already recorded task
           
            else: # this is a user (test) task
                fb = record_time(device, 'start', 'forward', task.task_id, timing_info)
                self.task_trace[nodeID][stageID]['fb'] = fb
                if task.task_id in self.all_trace[nodeID]: # this task has been completely recorded
                    self.all_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task
                    
                if stageID == 0: # first stage
                    self.user_task_record[task.task_id]['start'] = fb
                    
                with torch.no_grad():
                    tuple_outputs = self.distributed_stages[nodeID][stageID](**inputs, labels=task.feedback)
                # torch.cuda.synchronize() # synchronize the device
                fe = record_time(device, 'end', 'forward', task.task_id, timing_info)
                
                self.task_trace[nodeID][stageID]['fe'] = fe
                if task.task_id in self.all_trace[nodeID]: # this task has been completely recorded
                    self.all_trace[nodeID][task.task_id][stageID]['fe'] = fe # update the already recorded task
                
                if stageID == self.num_gpus_per_node - 1: # last stage
                    self.user_task_record[task.task_id]['end'] = fe
                    
            if self.eta is None:
                self.forward_end.record()
                torch.cuda.synchronize()
                # Profile forward time (seconds) per stage
                self.eta = self.forward_start.elapsed_time(self.forward_end) / (task.query['input_ids'].shape[1] ** 2 * 1e3)
            
        except Exception as e:
            logging.error(f"[Node {nodeID} - stage {stageID} - device {device}] Forward error occurred: {e}")
            tuple_outputs = None
        
        return tuple_outputs
    
    
    def calculate_node_LH(
        self, 
        nodeID: int,
        task_length: int,
        window_size: int = 10,
    ):
        history_lengths = [task['length'] for task in self.node_allocation[nodeID][-window_size:]]
        if len(history_lengths) == 0:
            return 0
        mean, std = np.mean(history_lengths), np.std(history_lengths)
        if std == 0:
            return 0
        # Gaussian probability based on task length as length heterogeneity
        return scipy.stats.norm(mean, std).pdf(task_length)
        
    
    def selective_algorithm(
        self, 
        task_length: int,
        training_step: int,
        trained_task_lengths: List[int],
        total_training_tasks: int,
    ):
        # print("Selective algorithm will be implemented!")
        # if not trained_task_lengths:
        #     return True
        # length_mean, length_std = np.mean(trained_task_lengths), np.std(trained_task_lengths)
        # if length_std == 0:
        #     return True
        try:
            # Gaussian probability based on task length
            # P_l = (1 / (self.std_length * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((task_length - self.mean_length) / self.std_length) ** 2)
            P_l = scipy.stats.norm(self.mean_length, self.std_length).pdf(task_length) # scipy probability density function
            # Adjusting probability based on training progress
            P_adjusted = 0.1 * P_l + 0.9 * (1 - (training_step / total_training_tasks))
            # P_adjusted = 1 - (training_step / total_training_tasks)
            print("Task length: {}, Gaussian P: {:.4f} -> P_adjusted: {:.4f}".format(task_length, P_l, P_adjusted))
            # print("Task length: {}, P_adjusted: {:.4f}".format(task_length, P_adjusted))
            do_backward = P_adjusted > np.random.random()
        except Exception as e:
            print(f"Error in selective algorithm: {e}")
            do_backward = True
        
        return do_backward
    

    def device_inference(
        self,
        stageID: int,
        nodeID: int,
        timing_info: Dict[str, List[float]],
        preloaded_tasks: List[Task], 
        deviceQueue: Union[queue.Queue, queue.PriorityQueue],
        nextdeviceQueue: Optional[queue.Queue] = None,
        init_device: Optional[int] = None,
    ):
        raise NotImplementedError("device_inference method must be implemented")             


    def node_inference(
        self,
        nodeID: int,
        node: Node,
    ):
        # We use num_gpus_per_node workers to simulateously get task from the queue and inference
        with ThreadPoolExecutor(max_workers=self.num_gpus_per_node) as executor:
            # futures = []
            for stageID in range(self.num_gpus_per_node):
                future = executor.submit(
                    self.device_inference, 
                    stageID,
                    nodeID,
                    self.timing_infos[nodeID], 
                    self.distributed_preloaded_tasks[nodeID],
                    node.device_queues[stageID],
                    nextdeviceQueue=node.device_queues[stageID+1] if stageID != len(self.distributed_stages[nodeID]) - 1 else None,
                    init_device=node.init_device,
                )
            #     futures.append(future)
            # for future in futures:
            #     try:
            #         # Set a timeout for each task. Adjust the timeout value as needed.
            #         future.result(timeout=60)  # Timeout set to 60 seconds
            #     except TimeoutError:
            #         # Handle the timeout, for example, by logging an error, retrying the task, or skipping it.
            #         print(f"Task execution exceeded the timeout limit and was aborted. Node ID: {nodeID}")
                
        print("Node {} finished inference".format(node.node_id))


    def run_stages_concurrently(self):
        with ThreadPoolExecutor(max_workers=len(self.distributed_nodes)) as executor:
            for nodeID, node in self.distributed_nodes.items():
                future = executor.submit(
                    self.node_inference, 
                    nodeID,
                    node, 
                )
            
            
    def run(self):
        # Run the stages concurrently
        task_queue = queue.Queue()
        with ThreadPoolExecutor(max_workers=3) as executor:
            future1 = executor.submit(
                self.producer,
                task_queue, 
            )
            future2 = executor.submit(
                self.globalScheduler,
                task_queue,
            )
            future3 = executor.submit(self.run_stages_concurrently)
        
        # Delete checkpoint file in the disk if self.ckpt_path is not None
        if self.ckpt_path is not None:
            for j in range(self.num_gpus_per_node):
                if os.path.exists(f"{self.ckpt_path}_stage{j}.pt"):
                    os.remove(f"{self.ckpt_path}_stage{j}.pt")
        
        # Save timing info
        self.save_timing_info()
        
        # Calculate metrics
        self.calculate_metrics()
        
        
    def save_timing_info(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.setting == 'isolated':
            setting = f"isolated-split{self.isolated_split}"
        else:
            setting = self.setting
        
        length_heterogeneity = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
        active_selection = f"active{self.active_selection}" if self.active_selection is not None else "active_1.0"
        # task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta})" if self.task_assignment == 'workload' else self.task_assignment
        for nodeID, timing_info in self.timing_infos.items():
            # # Remove the first start and end time for each GPU
            # gpus = list(set(int(key.split('_')[0]) for key in timing_info))
            # for gpu_id in gpus:
            #     timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start']
            #     timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end']
            stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{self.task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_node{nodeID}.json'
            with open(stats_f, 'w') as f:
                json.dump(timing_info, f, indent=4)
        
        
    def calculate_metrics(
        self, 
        metrics: Optional[Dict[str, Union[float, int]]] = None,
    ):
        metrics = metrics if metrics is not None else self.metrics
        if self.setting == 'isolated':
            setting = f"isolated-split{self.isolated_split}"
        else:
            setting = self.setting
            
        # Calculate metrics
        global_min_time, global_max_time = float('inf'), float('-inf')
        total_idles = []
        total_latencies = []
        node_idles = defaultdict(list)
        node_latencies = defaultdict(list)
        node_timelines = {}
        for nodeID, node in self.distributed_nodes.items():
            timing_info = {k: [[t[0], t[1]] for t in v] for k, v in self.timing_infos[nodeID].items()}
            if not timing_info:
                continue
            
            node_min_time, node_max_time = float('inf'), float('-inf')
            for gpu_id in range(self.num_gpus_per_node):
                min_t, max_t = float('inf'), float('-inf')
                gpu_idx = node.init_device + gpu_id
                starts = timing_info.get(f"{gpu_idx}_start", [])
                ends = timing_info.get(f"{gpu_idx}_end", [])
                if len(starts) == 1:
                    idles = []
                else:
                    idles = [start - end for (start, _), (end, _) in zip(starts[1:], ends[:-1]) if (start > end)]
                total_idles.extend(idles)
                node_idles[nodeID].extend(idles)
                
                tasks = list(zip(starts, ends))
                for i, ((start, start_label), (end, _)) in enumerate(tasks):
                    metrics[start_label].append(end - start)
                    min_t = min(min_t, start)
                    max_t = max(max_t, end)
                total_latencies.append(max_t - min_t)
                node_latencies[nodeID].append(max_t - min_t)
                global_min_time = min(global_min_time, min_t)
                global_max_time = max(global_max_time, max_t)
                node_min_time = min(node_min_time, min_t)
                node_max_time = max(node_max_time, max_t)
                node_timelines[nodeID] = (node_min_time, node_max_time)
                    
        bubble_rate = sum(total_idles) / sum(total_latencies) if sum(total_latencies) > 0 else 0
        for key, value in metrics.items():
            if key == 'loss':
                losses = value
            metrics[key] = sum(value) / len(value)
        
        # Calculate response times
        metrics['num_tasks'] = self.total_tasks
        metrics['retrain_tasks'] = self.retraining_tasks
        metrics['actual_retrained_tasks'] = len(self._trained_task_lengths)
        metrics['user_tasks'] = len(self.user_task_record)
        metrics['bubble_rate'] = bubble_rate 
        metrics['end2end_latency'] = global_max_time - global_min_time
        metrics['throughput'] = self.total_tasks / (global_max_time - global_min_time)
        metrics['num_tasks (node)'] = {nodeID: len(self.node_allocation[nodeID]) for nodeID in self.node_allocation}
        metrics['retrain_tasks (node)'] = {nodeID: len([task for task in self.node_allocation[nodeID] if task['type'] == 'training']) for nodeID in self.node_allocation}
        metrics['user_tasks (node)'] = {nodeID: len([task for task in self.node_allocation[nodeID] if task['type'] == 'test']) for nodeID in self.node_allocation}
        metrics['bubble_rate (node)'] = {
            nodeID: sum(idles) / sum(latencies) if sum(latencies) > 0 else 0 
            for nodeID, idles, latencies in zip(node_idles.keys(), node_idles.values(), node_latencies.values())
        }
        metrics['end2end_latency (node)'] = {nodeID: node_timelines[nodeID][1] - node_timelines[nodeID][0] for nodeID in node_timelines}
        metrics['throughput (node)'] = {nodeID: len(self.node_allocation[nodeID]) / (node_timelines[nodeID][1] - node_timelines[nodeID][0]) for nodeID in node_timelines}
        metrics['idles_sum'] = sum(total_idles)
        metrics['idles_sum (node)'] = {nodeID: sum(idles) for nodeID, idles in node_idles.items()}
        metrics['idles_avg'] = sum(total_idles) / len(total_idles)
        metrics['idles_avg (node)'] = {nodeID: sum(idles) / len(idles) if len(idles) > 0 else 0 for nodeID, idles in node_idles.items()}
        metrics['length_statistics'] = {
            'mean': self.mean_length,
            'std': self.std_length,
            'medium': self.medium_length,
            'min': self.min_length,
            'max': self.max_length,
        }
        
        if self.user_task_record:
            # total_response_time, total_wait_time, total_inference_time = 0, 0, 0
            response_times, wait_times, latencies = [], [], []
            user_global_min_time, user_global_max_time = float('inf'), float('-inf')
            for taskID, record_dict in self.user_task_record.items():
                # total_response_time += record_dict['end'] - record_dict['release']
                # total_wait_time += record_dict['start'] - record_dict['release']
                # total_inference_time += record_dict['end'] - record_dict['start']
                user_global_min_time = min(user_global_min_time, record_dict['start'])
                user_global_max_time = max(user_global_max_time, record_dict['end'])
                response_times.append(record_dict['end'] - record_dict['release'])
                wait_times.append(record_dict['start'] - record_dict['release'])
                latencies.append(record_dict['end'] - record_dict['start'])
                
            metrics['user_wait_avg'] = sum(wait_times) / len(self.user_task_record)
            metrics['user_inference_avg'] = sum(latencies) / len(self.user_task_record)
            metrics['user_response_avg'] = sum(response_times) / len(self.user_task_record)
            metrics['user_end2end_latency'] = user_global_max_time - user_global_min_time
            metrics['user_throughput'] = len(self.user_task_record) / (user_global_max_time - user_global_min_time)
            metrics['user_responses'] = response_times # list
            
        metrics['idles'] = total_idles # list
        metrics['losses'] = losses # list    
        
        # Save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        length_heterogeneity = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
        active_selection = f"active{self.active_selection}" if self.active_selection is not None else "active_1.0"
        # task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta})" if self.task_assignment == 'workload' else self.task_assignment
        stats_f = f'{self.output_dir}/metrics_{self.model_n}_{self.task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}.json'
        with open(stats_f, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {stats_f}")

    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, help='model name or path')
    parser.add_argument('--model_name', type=str, default='dummy', help='model name')
    parser.add_argument('--memory_threshold', type=float, default=0.5, 
                        help='threshold for maximum memory allocation in each GPU device')
    parser.add_argument('--access_token', type=str, default=None, help='access token')
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_length', action='store_true', help='save the length of each task')
    parser.add_argument('--setting', type=str, default='active', choices=['active','interval','isolated'], 
                        help='training setting')
    parser.add_argument('--isolated_split', type=float, default=None, 
                        help='split ratio for isolated test & train nodes. If not provided, the retraining rate is used.')
    parser.add_argument('--priority', type=str, default='FIFO', help='scheduling priority, default: FIFO')
    parser.add_argument('--task_assignment', type=str, default='random', choices=['rr', 'random', 'workload', 'util'], 
                        help='node level scheduling policy')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1, help='proportion of training tasks')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--train_lambda', type=int, default=10, help='Average number of training tasks produced per second')
    parser.add_argument('--test_lambda', type=int, default=10, help='Average number of test tasks produced per second')
    parser.add_argument('--alpha', type=float, default=1, help='response time coefficient')
    parser.add_argument('--beta', type=float, default=1, help='length heterogeneity coefficient')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='small value to avoid division by zero')
    parser.add_argument('--workload', type=str, default='poisson', help='workload arrival pattern')
    parser.add_argument('--length_distribution', type=str, default='random', choices=['random', 'ascending', 'descending', 'bursty'], 
                        help='distribution of input sequence length')
    parser.add_argument('--length_heterogeneity', type=int, default=None, 
                        help='standard deviation of the length distribution of the sampled subset')
    parser.add_argument('--active_selection', type=str, default=None,
                        help='active selection ratio for training tasks')
    parser.add_argument('--output_dir', type=str, default='prof')
    args = parser.parse_args()
    
    distributed_llm = DistributedLLM(args)
    distributed_llm.run()
