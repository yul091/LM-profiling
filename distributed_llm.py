import os
import sys
sys.dont_write_bytecode = True
import time
import json
import queue
import random
import argparse
import logging
import numpy as np
import gc
import GPUtil
from GPUtil import GPU
from typing import List, Dict, Optional, Any, Union, Tuple, Callable
from collections import defaultdict
from memory_profiler import profile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    set_seed,
    get_scheduler,
)
from utils import Node, Task, record_time, log_queue_contents, save_metrics_with_order
from models import (
    get_stages, 
    _prepare_inputs,
    _prepare_decoding_inputs,
)


# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run_experiment(args: argparse.Namespace, experimentID: int):
    print(f"\n ** Experiment {experimentID+1} **\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize and run the distributed model
    distributed_llm = DistributedLLM(args, experimentID=experimentID)
    distributed_llm.run()
    record_mode = distributed_llm.RECORD_MODE
    
    # Clean up resources explicitly
    del distributed_llm
    torch.cuda.empty_cache()
    gc.collect()

    # Rerun if necessary based on specific conditions
    if record_mode and args.run_mode == 'online':  # Assuming record_mode is a valid arg
        new_run = DistributedLLM(args, experimentID=experimentID)
        new_run.run()
        
        # Final clean up
        del new_run
        torch.cuda.empty_cache()
        gc.collect()
        
 

class DistributedLLM:

    def __init__(self, args: argparse.Namespace, experimentID: int = 0):
        self.n_samples = args.n_samples
        self.setting = args.setting
        self.priority = args.priority
        self.num_nodes = args.num_nodes
        self.num_gpus = args.num_gpus
        self.batch_size = args.batch_size
        self.rate_lambda = args.rate_lambda
        self.output_dir = args.output_dir
        self.task_assignment = args.task_assignment
        self.dataset_name_or_path = args.dataset_name_or_path
        self.lr = args.lr
        self.workload = args.workload
        self.PP = args.PP
        self.retraining_rate = args.retraining_rate  
        self.model_n = args.model_name
        self.save_length = args.save_length
        self.length_distribution = args.length_distribution
        self.length_heterogeneity = args.length_heterogeneity
        self.active_selection = args.active_selection
        self.profile_dir = args.profile_dir
        self.experimentID = experimentID
        self.run_mode = args.run_mode
        self.no_prior_profile = args.no_prior_profile
        self.no_memory_check = args.no_memory_check
        self.no_prioritization = args.no_prioritization
        self.max_wait = args.max_wait
        
        os.makedirs(self.profile_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # Count number GPUs per node
        if self.num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            # Ensure the user-specified number of GPUs does not exceed the available GPUs
            available_gpus = torch.cuda.device_count()
            if self.num_gpus > available_gpus:
                raise ValueError(f"Requested {self.num_gpus} GPUs, but only {available_gpus} GPUs are available.")
        self.num_gpus_per_node = self.num_gpus // self.num_nodes

        self.training_status = {}  # taskID: True/False (finished or not)
        self.deprioritized_tasks = queue.Queue()

        lh = 'default' if self.length_heterogeneity is None else self.length_heterogeneity
        self.profile_file = f"{self.profile_dir}/{self.model_n}_lambda={self.rate_lambda}_nodes={self.num_nodes}_retrain={self.retraining_rate}_batch={self.batch_size}.json"
        self.RECORD_MODE = False if self.run_mode == 'online' else True
        if self.task_assignment == 'workload':
            if self.profile_file is None or not os.path.exists(self.profile_file):
                self.RECORD_MODE = True
        
        self.slo_threshold = 0.114 * self.num_gpus_per_node  # SLO threshold (the forward time)
        if self.RECORD_MODE:
            self.retraining_rate = 0.5
            self.task_assignment = 'rr'
            self.active_selection = None
            self.froward_eta, self.backward_eta = None, None
            self.record_dict = defaultdict(list)
        else:
            if self.no_prior_profile or self.task_assignment != 'workload':
                self.froward_eta, self.backward_eta = None, None
                print("\n ** No offline profile used [pure ONLINE] **\n")
            else:
                file = json.load(open(self.profile_file))
                self.froward_eta = file['forward_eta'] if 'forward_eta' in file else None # forward coefficient
                self.backward_eta = file['backward_eta'] if 'backward_eta' in file else None # backward latency
                self.slo_threshold = file['FT'] * self.num_gpus_per_node if 'FT' in file else None # SLO threshold (5x the forward time)
                # self.profiled_losses = file['loss'] # list of offline training losses
                self.profiled_loss_dict = {int(taskID): float(instance['loss']) for (taskID, instance) in file['task_dict'].items()} # {taskID: loss}
                print("\n ** Offline profile loaded: forward_eta={}, backward_eta={} **\n".format(self.froward_eta, self.backward_eta))
        
        self.alpha = args.alpha
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.k = args.k
            
        # If dynamic method has decided the number of nodes, use that for 'rr+', 'random+', 'util+'
        if self.setting == 'isolated':
            self.isolated_split = args.isolated_split if args.isolated_split is not None else self.retraining_rate
            setting = f"isolated-split{self.isolated_split}"
        else:
            setting = self.setting
            
        self.used_nodes = self.num_nodes
        if '+' in self.task_assignment: # strong baselines (e.g., rr+, random+, util+)
            lh = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
            asl = f"active_{self.active_selection}" if self.active_selection is not None else "active_1.0"
            task_assignment = f"workload(a={self.alpha}|b={self.beta}|tau={self.epsilon})"
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{lh}-{self.length_distribution}_{self.retraining_rate}-{asl}_ID=0.json'
            
            if not os.path.exists(stats_f):
                raise ValueError(f'Cannot find dynamic result: {stats_f}')
            workload_res = json.load(open(stats_f))
            self.used_nodes = sum(x > 0 for x in workload_res["num_tasks (node)"].values())
            print(f"\n ** TASK ASSIGNMENT: {self.task_assignment} | Number of used nodes: {self.used_nodes} (out of {args.num_nodes}) **\n")
            
        if self.setting == 'isolated':
            if self.isolated_split != 100:
                num_train_nodes = max(1, round(self.used_nodes * self.isolated_split))
                if self.retraining_rate == 0:
                    num_train_nodes = 0
                num_test_nodes = max(1, self.used_nodes - num_train_nodes)
                if self.retraining_rate == 1:
                    num_test_nodes = 0
            else:
                num_test_nodes = self.used_nodes // 4
            
            self._test_nodes = list(range(num_test_nodes))
            self._train_nodes = list(range(num_test_nodes, self.used_nodes))
            print(f"** ISOLATED SYSTEM: Test nodes: {self._test_nodes}, Train nodes: {self._train_nodes} **")
        else:
            self._train_nodes = list(range(self.used_nodes))
            self._test_nodes = list(range(self.used_nodes))
        
        if self.priority is not None:
            self.ckpt_path = f'{self.output_dir}/stages-{self.task_assignment}_{self.model_n}_{setting}-{self.priority}_{self.workload}_{self.retraining_rate}'
        else:
            self.ckpt_path = f'{self.output_dir}/stages-{self.task_assignment}_{self.model_n}_{setting}_{self.workload}_{self.retraining_rate}'

        self.task_arrival = defaultdict(dict)
        self.task_trace = defaultdict(lambda: defaultdict(dict))
        self.train_trace = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.user_task_record = defaultdict(dict)
        self.all_trace = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.node_accumulated_bubble = defaultdict(int)
        self.forward_start = torch.cuda.Event(enable_timing=True)
        self.forward_end = torch.cuda.Event(enable_timing=True)
        self.backward_start = torch.cuda.Event(enable_timing=True)
        self.backward_end = torch.cuda.Event(enable_timing=True)

        # Define node instance
        self.distributed_nodes = {
            nodeID: Node(
                nodeID, 
                self.num_gpus_per_node, 
                init_device=nodeID * self.num_gpus_per_node,
            ) for nodeID in range(self.num_nodes)
        }
        self.memory_threshold = args.memory_threshold
        self.device_total_memory = torch.cuda.get_device_properties(0).total_memory
        self.timing_infos = {nodeID: defaultdict(list) for nodeID in range(self.num_nodes)}
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
        
        # Load datasets and dataloaders
        datasets = load_dataset(self.dataset_name_or_path)
        if self.RECORD_MODE:
            dataset = datasets['train']
        else:
            dataset = datasets['test']

        def tokenize_and_align_labels(examples):
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
        
        # print("Dataset example (first 2 examples):", dataset[:2])
        dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns(dataset.column_names)
        
        # Do sampling according to the length distribution
        input_lengths = [len(x) for x in dataset['input_ids']]
        self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length \
            = np.mean(input_lengths), np.std(input_lengths), np.median(input_lengths), min(input_lengths), max(input_lengths)
        print(" ** Original data length distribution: mean={}, std={}, medium={}, min={}, max={} **".format(
            self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length))
        if self.n_samples > 0:
            n_samples = min(self.n_samples, len(input_lengths))
            if self.length_heterogeneity is None:
                indices = random.sample(range(len(input_lengths)), n_samples)
                dataset = dataset.select(indices)
            else:
                indices = self._sample_subset_indices(input_lengths, n_samples, self.mean_length, self.length_heterogeneity)
                dataset = dataset.select(indices)
  
            subset_lengths = [len(x) for x in dataset['input_ids']]
            self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length \
                = np.mean(subset_lengths), np.std(subset_lengths), np.median(subset_lengths), min(subset_lengths), max(subset_lengths)
            print(f" ** Sampled {len(subset_lengths)} data points: mean={self.mean_length}, std={self.std_length}, medium={self.medium_length}, min={self.min_length}, max={self.max_length} **")
        
        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=None,
        )
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )
        
        # Preloaded dataset
        self.distributed_preloaded_tasks, self.training_taskIDs, self.inference_taskIDs = self.get_preloaded_dataset(
            self.distributed_nodes, 
            self.dataloader, 
            retraining_rate=self.retraining_rate,
        )
        self.total_tasks = len(self.training_taskIDs) + len(self.inference_taskIDs)
        self.retraining_tasks = len(self.training_taskIDs)
        self.saving_steps = max(min(100, self.retraining_tasks // 2), 1)
        print(f" ** Total tasks: {self.total_tasks}, retraining tasks: {self.retraining_tasks}, inference tasks: {len(self.inference_taskIDs)}, saving steps: {self.saving_steps} ** \n")
        self._training_step = 0
        self._trained_task_lengths = []
        
        # Select tasks to be trained
        if (self.active_selection is None) or self.RECORD_MODE: 
            self.backward_taskIDs = set(self.training_taskIDs)
        elif self.active_selection in ['random', 'first', 'last']:
            lh = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
            adapt_f = f'{self.output_dir}/metrics_{self.model_n}_{self.task_assignment}_{setting}-{self.priority}_{self.workload}-{lh}-{self.length_distribution}_{self.retraining_rate}-active_adaptive_ID=0.json'
            if os.path.exists(adapt_f):
                adapt_res = json.load(open(adapt_f))
                actual_trained_tasks = adapt_res['actual_retrained_tasks']
                print(f"\n ** Actual tasks to be trained: {actual_trained_tasks} (according to adaptive) **\n")
                if self.active_selection == 'random':
                    self.backward_taskIDs = set(random.sample(self.training_taskIDs, k=actual_trained_tasks))
                elif self.active_selection == 'first':
                    self.backward_taskIDs = set(self.training_taskIDs[:actual_trained_tasks])
                else:
                    self.backward_taskIDs = set(self.training_taskIDs[-actual_trained_tasks:])
            else:
                self.backward_taskIDs = set(self.training_taskIDs)
            
        # Save task length distribution for further analysis
        if self.save_length:
            length_dict = {taskID: task.query['input_ids'].shape[1] for taskID, task in enumerate(self.distributed_preloaded_tasks[0])}
            with open(f"{self.output_dir}/task_length_{self.model_n}_{setting}_{self.workload}_{self.retraining_rate}.json", 'w') as f:
                json.dump(length_dict, f, indent=4)

        # Stages
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
    

    def offload_to_cpu(self, tensor: Any):
        """
        Offload tensor to CPU to free GPU memory.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.to("cpu", non_blocking=True)
        elif isinstance(tensor, dict):  # Handles customized outputs
            return {k: self.offload_to_cpu(v) for k, v in tensor.items()}
        elif isinstance(tensor, list):
            return [self.offload_to_cpu(v) for v in tensor]
        elif isinstance(tensor, tuple):
            return tuple(self.offload_to_cpu(v) for v in tensor)
        return tensor

    def reload_to_gpu(self, tensor: Any, device: torch.device):
        """
        Reload tensor from CPU back to GPU.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.to(device, non_blocking=True)
        elif isinstance(tensor, dict):
            return {k: self.reload_to_gpu(v, device) for k, v in tensor.items()}
        elif isinstance(tensor, list):
            return [self.reload_to_gpu(v, device) for v in tensor]
        elif isinstance(tensor, tuple):
            return tuple(self.reload_to_gpu(v, device) for v in tensor)
        return tensor

        
    def get_preloaded_dataset(
        self,
        distributed_nodes: Optional[Dict[int, Node]] = None, 
        dataloader: Optional[DataLoader] = None, 
        retraining_rate: Optional[float] = None,
    ) -> Tuple[Dict[int, List[Task]], List[int], List[int]]:
        
        print("Using preloaded data ...")
        distributed_nodes = distributed_nodes if distributed_nodes is not None else self.distributed_nodes
        dataloader = dataloader if dataloader is not None else self.dataloader
        retraining_rate = retraining_rate if retraining_rate is not None else self.retraining_rate
        distributed_preloaded_tasks = defaultdict(list)
        inference_taskIDs, training_taskIDs = [], []
        
        selected_data = []
        for i, batch in enumerate(dataloader):
            seq_length = batch['input_ids'].shape[1]
            # print(f"task {i}: {batch['input_ids'].shape} range ({batch['input_ids'].min()}, {batch['input_ids'].max()})")
            if batch['input_ids'].max() >= self.tokenizer.vocab_size:
                raise ValueError(f"Index error of token ID {batch['input_ids'].max()}")
            selected_data.append((seq_length, batch))
        inference_tasks = int(len(selected_data) * (1-retraining_rate))
        inference_ids = random.sample(range(len(selected_data)), k=inference_tasks)
            
        # Define the order of arrival input sequence length
        if self.length_distribution == 'ascending':
            selected_data.sort(key=lambda x: x[0])
        elif self.length_distribution == 'descending':
            selected_data.sort(key=lambda x: x[0], reverse=True)
        elif self.length_distribution == 'bursty':  # one short one long, ...
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
        
        
        # If workload is 'alternate', we need to create a list of varying lambda values (e.g., 1, 1, ..., 5, 5, ..., 30, 30, ...), each for X consecutive tasks
        if self.rate_lambda == -1 and retraining_rate != 1:
            print(f"\n ** Frequency-alternated {inference_tasks} serving requests **\n")
            lambda_values = [6, 12, 29, 15, 8, 20, 11, 6, 24, 19, 30, 14]
            total_lambda = sum(lambda_values)
            k = inference_tasks / total_lambda  # Proportionality constant

            # Initialize variables
            tasks_per_lambda_dict = {}
            total_assigned_tasks = 0
            inference_lambdas = []

            # Calculate tasks per lambda proportional to lambda values
            for lam in lambda_values:
                tasks = int(k * lam)
                tasks_per_lambda_dict[lam] = tasks
                total_assigned_tasks += tasks

            # Adjust for any remaining tasks due to integer rounding
            remaining_tasks = inference_tasks - total_assigned_tasks
            if remaining_tasks > 0:
                # Distribute the remaining tasks starting from the largest lambda
                sorted_lambdas = sorted(lambda_values, reverse=True)
                idx = 0
                while remaining_tasks > 0:
                    lam = sorted_lambdas[idx % len(sorted_lambdas)]
                    tasks_per_lambda_dict[lam] += 1
                    remaining_tasks -= 1
                    idx += 1

            # Construct a list of dictionaries to store ranges and node requirements
            lambda_ranges = []
            current_start_id = 0  # Initial task ID
            for lam in lambda_values:
                repetitions = tasks_per_lambda_dict[lam]
                inference_lambdas.extend([lam] * repetitions)
                test_nodes = self.num_nodes // 4 if lam <= 15 else self.num_nodes // 2
                current_end_id = current_start_id + repetitions - 1  # Calculate the end ID for this lambda range
                
                # Append the range to lambda_ranges
                lambda_ranges.append({
                    'lambda': lam,
                    'start_id': current_start_id,
                    'end_id': current_end_id,
                    'test_nodes': test_nodes
                })
                
                # Update the start ID for the next lambda
                current_start_id = current_end_id + 1

            if self.setting == 'isolated' and self.isolated_split == 100:
                # Now lambda_ranges contains the ranges and test node requirements
                lambda_ranges[-1]['end_id'] = len(selected_data) - 1
                self.ID2test = {}
                for taskID in range(len(selected_data)):
                    matched = False
                    for entry in lambda_ranges:
                        if entry['start_id'] <= taskID <= entry['end_id']:
                            self.ID2test[taskID] = entry['test_nodes']
                            matched = True
                            break
                    if not matched:
                        raise ValueError(f'No lambda intervals match task {taskID}')
                print(f'\nLambda Ranges and Test Node Requirements: {lambda_ranges}')


        # print(f'inference lambdas {len(inference_lambdas)}, inference_ids {len(inference_ids)}')
        j = 0
        for i, (_, batch) in enumerate(selected_data):
            # 10% of the time, produce a task with feedback
            require_training = i not in set(inference_ids)
            if require_training: 
                training_taskIDs.append(i)
                lamda = self.rate_lambda if self.rate_lambda != -1 else 5
            else:
                inference_taskIDs.append(i)
                lamda = self.rate_lambda if self.rate_lambda != -1 else inference_lambdas[j]
                j += 1

            # Prepare decode input
            inputs = _prepare_decoding_inputs(batch)
            for nodeID, node in distributed_nodes.items():
                task = Task(
                    task_id=i,
                    rate_lambda=lamda,
                    query=_prepare_inputs(inputs, device=node.init_device),
                    feedback=_prepare_inputs(inputs['labels'], device=node.last_device),
                    node_id=nodeID,
                    num_gpus_per_node=node.num_gpus_per_node,
                    require_training=require_training,
                )
                distributed_preloaded_tasks[nodeID].append(task)
        
        return distributed_preloaded_tasks, training_taskIDs, inference_taskIDs


    def serving_producer(
        self,
        taskQueue: queue.Queue, 
    ) -> None:
        """
        Simulate online serving requests with a Poisson arrival distribution.
        """
        # Produce using the dataset
        IDs = self.inference_taskIDs[::-1]
        while True:
            if not IDs:  # No more tasks to serve: terminate
                taskQueue.put(None)  # Signal the end of the dataset
                print(f"Producer finished producing inference tasks!")
                break

            taskID = IDs.pop()  # Pop the last task ID

            if self.workload == 'all':
                time.sleep(0)
            else:
                time.sleep(random.expovariate(self.distributed_preloaded_tasks[0][taskID].rate_lambda))
            
            taskQueue.put(taskID) 
            release = time.time()
            self.task_arrival[taskID]['release'] = release
            self.user_task_record[taskID]['release'] = release
            # print(f"Inference task {taskID} released at {release}")



    def training_producer(
        self,
        taskQueue: queue.Queue, 
        max_wait_time: int = 10,
    ) -> None:
        """
        Submit training tasks based on when the first-stage forward of the preceding task completes.
        """
        # Produce using the dataset
        IDs = self.training_taskIDs[::-1]
        prev_ID = None
        while IDs or not self.deprioritized_tasks.empty():
            # log_queue_contents(taskQueue, nodeID=None, stageID=None)
            # if not IDs:
            #     taskQueue.put(None)  # Signal the end of the dataset
            #     print(f"Producer finished producing training tasks!")
            #     break
            if IDs and not self.deprioritized_tasks.empty():
                # Pick the task with the highest priority (smallest ID)
                taskID = IDs.pop() if IDs[-1] < self.deprioritized_tasks.queue[0] else self.deprioritized_tasks.get()
                # taskID = IDs.pop()  # Pop the last task ID
            elif IDs:
                taskID = IDs.pop()
            else:
                taskID = self.deprioritized_tasks.get()

            start_time = time.time()  # Start timing for timeout
            while prev_ID is not None and not self.training_status[prev_ID]:  # Wait for the preceding task to complete
                time.sleep(0.01)  # Sleep briefly to avoid busy waiting
                if time.time() - start_time > max_wait_time:
                    print(f"Timeout waiting for preceding training task {prev_ID} to complete.")
                    self.training_status[prev_ID] = True  # Force status update to avoid indefinite blocking
                    break
            
            # Proceed after preceding task has completed
            self.training_status[taskID] = False
            taskQueue.put(taskID)
            self.task_arrival[taskID]['release'] = time.time()
            # print(f"Training task {taskID} released at {self.task_arrival[taskID]['release']}.")
            prev_ID = taskID

        taskQueue.put(None)  # Signal the end of the dataset
        print(f"Producer finished producing training tasks!")

        # # Enqueue deprioritized tasks if any
        # if not self.deprioritized_tasks.empty():
        #     train_taskID = self.deprioritized_tasks.get()
        #     # taskQueue.put(train_taskID)
        #     # If the last element in the queue is None (mark the end), insert the train_taskID just before that
        #     if taskQueue.queue[-1] is None:
        #         taskQueue.queue.insert(-1, train_taskID)
        #     else:
        #         taskQueue.put(train_taskID)
        #     self.task_arrival[train_taskID]['release'] = time.time()
        
        
    # @profile 
    def _compute_priority(self, nodeID: int, taskID: int, do_backward: bool = False) -> float:
        """
        Predict the execution traces for the incoming task and compute the reward (bubble rate & response time).
        """
        batch_size, length = self.distributed_preloaded_tasks[nodeID][taskID].query['input_ids'].shape
        # Initialize bubble rate and response time
        fb = self.task_arrival[taskID]['release']  # arrival time
        II, init_id, finished_ids = 0, 0, []
        previous_taskIDs = list(self.all_trace[nodeID].keys())
        previous_train_taskIDs = list(self.train_trace[nodeID].keys())
        forward_eta = self.froward_eta if self.froward_eta is not None else 2e-6
        backward_eta = self.backward_eta if self.backward_eta is not None else 1e-6
        BL = backward_eta * batch_size * length ** 2  # backward latency
        FL = forward_eta * batch_size * length ** 2  # forward latency
            
        # Get/Estimate previous forward execution traces
        if previous_taskIDs:
            last_taskID = previous_taskIDs[-1]
            arrival_gap = self.task_arrival[taskID]['release'] - self.task_arrival[last_taskID]['release']
        else:
            last_taskID = None
            arrival_gap = 0
            
        for stageID in range(self.num_gpus_per_node):
            pfe = self.all_trace[nodeID][last_taskID][stageID]['fe'] if last_taskID is not None else fb  # previous forward end
            fb = max(pfe, fb)  # forward start
            fe = fb + FL  # forward end
            i = init_id  # initialize the index of the previous training tasks
            
            # Handling dependencies and inter-task interference
            offset = 0
            for i in range(init_id, len(previous_train_taskIDs)):
                ptID: int = previous_train_taskIDs[i]  # training task ID
                if fe <= self.train_trace[nodeID][ptID][stageID]['bb']:  # no interference
                    break
                fb = max(fb, self.train_trace[nodeID][ptID][stageID]['be'])  # far dependency
                fe = fb + FL

                if pfe <= self.train_trace[nodeID][ptID][stageID]['bb']:  # this training task operates in between the previous and current tasks
                    offset += backward_eta * batch_size * self.distributed_preloaded_tasks[nodeID][ptID].query['input_ids'].shape[1] ** 2

                if stageID == 0: # this training task has finished
                    finished_ids.append(ptID)
                    
            # Write the forward and backward execution traces
            self.task_trace[nodeID][stageID]['fb'] = fb
            self.task_trace[nodeID][stageID]['fe'] = fe
            II += max(fb - pfe - offset, 0)  # update II (subtract all backward latencies in between)
            init_id = i  # avoid recomputation in subsequent stages
        
        if do_backward:
            bb = fe # backward starts at the end of the forward
            for stageID in range(self.num_gpus_per_node-1, -1, -1):
                self.task_trace[nodeID][stageID]['bb'] = bb
                self.task_trace[nodeID][stageID]['be'] = bb + BL
                bb += BL
        
        # Clear the finished training tasks
        for ptID in finished_ids:
            self.train_trace[nodeID].pop(ptID)
            self.all_trace[nodeID].pop(ptID)
                    
        # Compute reward (idleness profit and response time)
        RT = fe - self.task_arrival[taskID]['release']  # response time (R)
        IP = -max(II/self.num_gpus_per_node - arrival_gap, self.epsilon)  # idleness profit (IP)
        LC = self.distributed_nodes[nodeID].length_consistency(length)  # length consistency (LC)
        priority = (IP + self.beta * LC) / (self.alpha * RT)
        # priority = IP / (self.alpha * RT + self.beta * LC + self.epsilon)
        
        return priority, RT
            
        
    # @profile
    def assign_node(self, node_list: List[int], taskID: int, do_backward: bool = False):
        
        if len(node_list) == 1:
            return node_list[0]
        if 'random' in self.task_assignment or self.RECORD_MODE: # Random assignment
            return random.choice(node_list)
        elif 'rr' in self.task_assignment: # Round-robin
            return node_list[taskID % len(node_list)]
        elif 'util' in self.task_assignment: # LUF: choose the node with the least average utilization across all its GPUs
            # gputils = self._get_gpu_utilization()
            # return min(node_list, key=lambda nodeID: gputils[nodeID])
            gpus: List[GPU] = GPUtil.getGPUs()
            return min(node_list, key=lambda nodeID: np.mean([gpus[device].memoryUtil for device in self.distributed_nodes[nodeID].device_ids]))
            # # return min(node_list, key=lambda nodeID: np.mean([torch.cuda.memory_allocated(device) for device in self.distributed_nodes[nodeID].device_ids]))
        elif self.task_assignment == 'workload':
            # # Choose the node with the least workload (number of tasks in the first device queue)
            # return min(node_list, key=lambda nodeID: self.distributed_nodes[nodeID].device_queues[0].qsize())
            priority_list, rt_list = [], []
            for nodeID in node_list:
                try:
                    priority, RT = self._compute_priority(nodeID, taskID, do_backward=do_backward)
                except Exception as e:
                    priority = random.random()
                    logging.error(f"[node {nodeID} | task {taskID}] Bubble calculation error occurred: {e}")
                priority_list.append(priority)
                rt_list.append(RT)
            
            # Greedy: assign task to the node with the highest reward (lowest average utilization)
            best_index = np.argmax(priority_list) 
            # make it more smart, if some reward values are the same, randomly choose one
            # best_index = np.random.choice(np.where(reward_list == np.max(reward_list))[0])
            # # Instead of greedy, let's use sampling with probability proportional to the reward
            # reward_list = F.softmax(torch.FloatTensor(reward_list), dim=0).numpy()  
            # best_index = np.random.choice(range(len(reward_list)), p=reward_list)
            nodeID = node_list[best_index]
            # rt = rt_list[best_index]
            # self.node_accumulated_bubble[nodeID] += bi_list[best_index] - br_list[best_index]
            return nodeID, rt_list, best_index 
        else:
            raise ValueError(f"Invalid task assignment method: {self.task_assignment}")
        

    # def estimate_response(self, trainID: int, nodeID: int = None) -> float:
    #     """
    #     Estimate the shortest response time of a task based on the current system state across all nodes.
    #     """
    #     # print(f"Estimating response time for current training task {trainID}...")
    #     forward_eta = self.froward_eta if self.froward_eta is not None else 2e-6
    #     # backward_eta = self.backward_eta if self.backward_eta is not None else 1e-6
    #     # batch_size, length = self.distributed_preloaded_tasks[nodeID][taskID].query['input_ids'].shape
    #     tb, tl = self.distributed_preloaded_tasks[nodeID][trainID].query['input_ids'].shape
    #     FL_train = forward_eta * tb * tl ** 2  # training forward latency
    #     # BL_train = backward_eta * tb * tl ** 2  # training backward latency
    #     arrival = self.task_arrival[trainID]['release']  # arrival time
    #     fb_train = arrival + FL_train * (self.num_gpus_per_node - 1)  # training forward start
    #     # FL_next = forward_eta * batch_size * length ** 2  # forward latency
    #     # arrival = self.task_arrival[taskID]['release'] # arrival time
    #     # fb_next = arrival + FL_next * (self.num_gpus_per_node - 1)  # forward start

    #     earlist_fe = float('inf')
    #     if nodeID is None:
    #         for nodeID in range(self.num_nodes):
    #             previous_taskIDs = list(self.all_trace[nodeID].keys())
    #             last_taskID = previous_taskIDs[-1] if previous_taskIDs else None
    #             if last_taskID is None:
    #                 earlist_fe = min(earlist_fe, fb_train + FL_train)
    #                 continue
    #             pfe = self.all_trace[nodeID][last_taskID][self.num_gpus_per_node - 1]['fe'] # previous forward end
    #             fe = max(pfe, fb_train) + FL_train  # forward end
    #             earlist_fe = min(earlist_fe, fe)
    #     else:
    #         previous_taskIDs = list(self.all_trace[nodeID].keys())
    #         last_taskID = previous_taskIDs[-1] if previous_taskIDs else None
    #         if last_taskID is None:
    #             return FL_train * self.num_gpus_per_node
    #         pfe = self.all_trace[nodeID][last_taskID][self.num_gpus_per_node - 1]['fe'] # previous forward end
    #         fe = max(pfe, fb_train) + FL_train  # forward end
    #         earlist_fe = min(earlist_fe, fe)

    #     return earlist_fe - arrival


        
    # @profile
    def globalScheduler(
        self, 
        taskQueue: queue.Queue, 
        max_wait_time: int = 10,
    ):
        finished_producers = 0  
        # Global scheduler
        while True:
            try:
                taskID: int = taskQueue.get(timeout=max_wait_time)  # Add a timeout to avoid indefinite blocking
                # print(f"Current global queue tasks {taskQueue.queue}")
            except queue.Empty:
                print("GlobalScheduler: No tasks in queue. Waiting...")
                continue

            if taskID is None:
                finished_producers += 1
                if finished_producers == self.producer_workers:  # All producers have finished
                    print("Global scheduler finished scheduling tasks")
                    for nodeID, node in self.distributed_nodes.items():
                        # Terminate each stage_inference by putting a larger number in the priority_queue
                        node.device_queues[0].put((float('inf'), float('inf')))
                        # Log queue contents after sending termination signal
                        log_queue_contents(node.device_queues[0], nodeID, stageID=0)
                    break
                continue
            
            # prioritization_start = time.time()
            # Active selection
            # select_start = time.time()
            # if self.distributed_preloaded_tasks[0][taskID].require_training:
            #     do_backward = self.check_do_backward(
            #         taskID, 
            #         self.distributed_preloaded_tasks[0][taskID].query['input_ids'].shape[1], 
            #         self._training_step, 
            #         self.retraining_tasks,
            #     ) if self.task_assignment == 'workload' else True
            # else:
            #     do_backward = False
            # self.metrics['active_selection'].append(time.time() - select_start)
            do_backward = self.distributed_preloaded_tasks[0][taskID].require_training

            # # If this task is a training task, do SLO check for the next inference task (only for LeMix)
            # if do_backward and self.task_assignment == 'workload':
            #     # Check for the next inference task in the queue
            #     next_inference_taskID = None
            #     with taskQueue.mutex:  # Safely access the queue
            #         for item in taskQueue.queue:
            #             if item and not self.distributed_preloaded_tasks[0][item].require_training:
            #                 next_inference_taskID = item
            #                 break
            #     # If there’s a next inference task, estimate its response time
            #     if next_inference_taskID is not None:
            #         estimated_response_time = self.estimate_response(next_inference_taskID)
            #         if estimated_response_time > self.slo_threshold:  # (5x forward latency)
            #             # SLO violation
            #             print(f"SLO violation detected for next inference task {next_inference_taskID}. Deprioritizing training task {taskID}.")
            #             # Reorder the queue: place the current training task right after the next inference task
            #             with taskQueue.mutex:  # Safely modify the queue
            #                 taskQueue.queue.insert(taskQueue.queue.index(next_inference_taskID) + 1, taskID)  # Place taskID after next_inference_taskID
            #             continue  # Skip further processing of this taskID for now
            
            # self.metrics['task_prioritization'].append(time.time() - prioritization_start)

            # Task assignment
            assign_start = time.time()
            if self.setting != 'isolated':
                if self.task_assignment == 'workload':
                    nodeID, RTs, best_index = self.assign_node(node_list=list(range(self.num_nodes)), taskID=taskID, do_backward=do_backward)
                else:
                    nodeID = self.assign_node(node_list=list(range(self.used_nodes)), taskID=taskID, do_backward=do_backward)
            else:
                ########################## Dynamic separate ########################
                test_nodes = self._test_nodes
                train_nodes = self._train_nodes
                if self.rate_lambda == -1 and self.isolated_split == 100:  # dynamic
                    num_test_nodes = self.ID2test[taskID]
                    test_nodes = list(range(num_test_nodes))
                    train_nodes = list(range(num_test_nodes, self.used_nodes))
                    print(f' ** Dynamic separate: taskID {taskID}, test_nodes {test_nodes}, train_nodes {train_nodes} ** ')
                ########################## Dynamic separate ########################
                
                if do_backward:  # assign to one of the training nodes
                    nodeID = self.assign_node(node_list=train_nodes, taskID=taskID, do_backward=do_backward)
                else: # assign to one of the test nodes
                    nodeID = self.assign_node(node_list=test_nodes, taskID=taskID, do_backward=do_backward)

            # Priotization based on the node states
            # If this task is a training task, do SLO check for the next inference task (only for LeMix)
            if self.task_assignment == 'workload' and (not self.no_prioritization) and do_backward and (finished_producers == 0):
                # estimated_response_time = self.estimate_response(taskID, nodeID=nodeID)
                if RTs[best_index] > 5 * self.slo_threshold:  # (5x SLO goal)
                    # SLO violation
                    # print(f"SLO violation detected for current training task {taskID}. Deprioritizing it in a temp queue {self.deprioritized_tasks.qsize()}.")
                    # Reorder the queue: place the current training task in a temporary queue
                    self.deprioritized_tasks.put(taskID)
                    self.training_status[taskID] = True  # Temporarily mark the task as completed to avoid indefinite blocking
                    continue  # Skip further processing of this taskID for now

            self.distributed_preloaded_tasks[nodeID][taskID].do_backward = do_backward
                    
            # Update train trace and all trace with the task trace
            if self.task_assignment == 'workload':
                self.all_trace[nodeID][taskID] = self.task_trace[nodeID].copy()
                if do_backward:
                    self.train_trace[nodeID][taskID] = self.task_trace[nodeID].copy()
                        
            self.metrics['node_assignment'].append(time.time() - assign_start)

            # # Each node queue store task IDs
            # while self.distributed_nodes[nodeID].device_queues[0].qsize() > 2:
            #     # print(f"Task {taskID} waiting due to queue limit on Node {nodeID}")
            #     time.sleep(0.01)  # Brief wait to avoid busy-waiting

            self.distributed_nodes[nodeID].device_queues[0].put((taskID, taskID))
            
            # Record the node allocation 
            self.distributed_nodes[nodeID].update_length_stats(
                length=self.distributed_preloaded_tasks[nodeID][taskID].query['input_ids'].shape[1],
                computation_type='training' if do_backward else 'test',
            )
            # print("Global scheduler scheduled task {} (requre_training={}) to node {}".format(taskID, self.distributed_preloaded_tasks[0][taskID].require_training, nodeID))
    
    
    def check_device_availability(self, device: int):
        """
        Check if the device has enough available memory.
        Args:
        - device: The device to check.
        Returns:
        - is_available: Boolean indicating if the device is available.
        """
        # Get device memory status
        allocated_memory = torch.cuda.memory_allocated(device)
        return allocated_memory / self.device_total_memory <= self.memory_threshold
        # available_memory = self.device_total_memory - allocated_memory
        # # Calculate the available memory ratio
        # available_ratio = available_memory / self.device_total_memory
        # # Check if the available memory ratio is above the threshold
        # return available_ratio > (1 - threshold)
    
    
    def wait_for_device_availability(
        self, 
        device: int, 
        check_interval: float = 0.1, 
        force_check: bool = True,
    ):
        """
        Wait until the device is available based on memory usage.
        Args:
        - device: The device to wait for.
        - check_interval: How often to check the device status (in seconds).
        """
        if self.no_memory_check and not force_check:
            return True
        start_time = time.time()
        while not self.check_device_availability(device):
            # print(f"Waiting for device {device} to become available...")
            time.sleep(check_interval)
            if time.time() - start_time > self.max_wait:
                print(f"Exceeded max wait time for device {device}. Exit forward waiting loop.")
                return False
        return True

            
    def forward(
        self, 
        task: Task,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        stageID: int,
        nodeID: int,
        device: int, 
        timing_info: Dict[str, List[float]],
    ) -> Tuple[torch.Tensor, ...]:
        # Memory check and scale down if necessary
        # self.wait_for_device_availability(device)

        try:
            if task.require_training: # this is a training task
                fb = record_time(device, 'start', 'forward_grad', task.task_id, timing_info)
                self.task_trace[nodeID][stageID]['fb'] = fb
                if task.task_id in self.train_trace[nodeID]: # this training task has been completely recorded
                    self.train_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task
                    self.all_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task

                # with autocast():
                tuple_outputs = self.distributed_stages[nodeID][stageID](**inputs, labels=task.feedback)
                fe = record_time(device, 'end', 'forward_grad', task.task_id, timing_info)

                if stageID == 0 and self.PP == 'async': # first stage: unlock the producer for the next task
                    self.training_status[task.task_id] = True  # Mark first-stage forward as completed
                
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
                    # with autocast():
                    tuple_outputs = self.distributed_stages[nodeID][stageID](**inputs, labels=task.feedback)
                fe = record_time(device, 'end', 'forward', task.task_id, timing_info)
                
                self.task_trace[nodeID][stageID]['fe'] = fe
                if task.task_id in self.all_trace[nodeID]: # this task has been completely recorded
                    self.all_trace[nodeID][task.task_id][stageID]['fe'] = fe # update the already recorded task
                
                if stageID == self.num_gpus_per_node - 1: # last stage
                    self.user_task_record[task.task_id]['end'] = fe
                    logging.info(f"[Node {nodeID}] Task {task.task_id} finished forward pass!")
                    
            if self.froward_eta is None:
                self.froward_eta = (fe - fb) / (task.query['input_ids'].shape[0] * task.query['input_ids'].shape[1] ** 2)
                    
            if self.RECORD_MODE:
                # Profile forward time (seconds) per stage
                self.record_dict['forward_etas'].append(
                    ((fe - fb) / (task.query['input_ids'].shape[0] * task.query['input_ids'].shape[1] ** 2), task.task_id)
                )
                self.record_dict['FTs'].append((fe - fb, task.task_id))
                
        except Exception as e:
            logging.error(f"[Node {nodeID} - stage {stageID} - device {device}] Forward error occurred: {e}")
            if stageID == 0  and self.PP == 'async':
                self.training_status[task.task_id] = True  # Mark first-stage forward as completed
            tuple_outputs = None
        
        return tuple_outputs
    

    def stage_inference(
        self,
        stageID: int,
        nodeID: int,
        timing_info: Dict[str, List[float]],
        preloaded_tasks: List[Task], 
        deviceQueue: Union[queue.Queue, queue.PriorityQueue],
        nextdeviceQueue: Optional[queue.Queue] = None,
        init_device: Optional[int] = None,
    ):
        raise NotImplementedError("stage_inference method must be implemented")             


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
                    self.stage_inference, 
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
            #         print(f"Task execution exceeded the timeout limit and was aborted for Node {nodeID}")
            #     except Exception as e:
            #         print(f"Exception occurred in Node {nodeID}, Stage {stageID}: {e}")
                
        print("Node {} finished inference".format(node.node_id))


    def execute_concurrently(self):
        with ThreadPoolExecutor(max_workers=len(self.distributed_nodes)) as executor:
            # futures = []
            for nodeID, node in self.distributed_nodes.items():
                executor.submit(self.node_inference, nodeID, node)
            #     futures.append(future)
            # for future in futures:
            #     try:
            #         future.result(timeout=60)  # Set timeout to avoid indefinite waits
            #     except TimeoutError:
            #         print(f"Task execution exceeded the timeout limit for Node {nodeID}.")


    def run(self):
        # Run the stages concurrently
        if self.RECORD_MODE:
            print("\n ** Running in RECORD mode **\n")
        
        task_queue = queue.Queue()
        self.producer_workers = 0  # One for the global scheduler and one for the executor
        if self.training_taskIDs:
            self.producer_workers += 1
        if self.inference_taskIDs:
            self.producer_workers += 1

        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with ThreadPoolExecutor(max_workers=self.producer_workers) as producer_executor, \
             ThreadPoolExecutor(max_workers=1) as scheduler_executor, \
             ThreadPoolExecutor(max_workers=1) as execution_executor:

            # Submit producers
            if self.inference_taskIDs:
                producer_executor.submit(self.serving_producer, task_queue)
            if self.training_taskIDs:
                producer_executor.submit(self.training_producer, task_queue)
            
            # Submit global scheduler
            scheduler_executor.submit(
                self.globalScheduler,
                task_queue,
            )
            # Submit execution tasks
            execution_executor.submit(self.execute_concurrently)
        
        if self.RECORD_MODE:        
            # Save recorded dict
            losses = [loss for loss, _ in self.record_dict['loss']]
            # Quantile (25, 50, 75) for record_dict['loss]
            self.record_dict['loss_stats'] = {'mean': np.mean(losses), 'std': np.std(losses)}
            for quantile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                self.record_dict['loss_stats'][f'{quantile}%'] = np.percentile(losses, quantile)
                
            loss_dict = {taskID: loss for (loss, taskID) in self.record_dict['loss']}
            length_dict = {taskID: length for (length, taskID) in self.record_dict['length']}
            # First get the forward dict where each taskID corresponds to multiple forward times
            tmp = defaultdict(list)
            for (ft, taskID) in self.record_dict['FTs']:
                tmp[taskID].append(ft)
            # Then take the sum of the forward times for each taskID
            forward_dict = {taskID: np.mean(fts) for taskID, fts in tmp.items()}
            forward_eta_dict = {taskID: et for et, taskID in self.record_dict['forward_etas']}
            backward_dict = {taskID: bt for bt, taskID in self.record_dict['BTs']}
            backward_eta_dict = {taskID: et for et, taskID in self.record_dict['backward_etas']}
            self.record_dict['forward_eta'] = np.mean(list(forward_eta_dict.values()))
            self.record_dict['backward_eta'] = np.mean(list(backward_eta_dict.values()))
            self.record_dict['FT'] = np.mean(list(forward_dict.values()))
            self.record_dict['BT'] = np.mean(list(backward_dict.values()))
            
            self.record_dict['task_dict'] = {
                taskID: {
                    'loss': loss_dict.get(taskID, 'None'),
                    'length': length_dict.get(taskID, 'None'),
                    'forward': forward_dict.get(taskID, 'None'),
                    'backward': backward_dict.get(taskID, 'None'),
                    'forward_eta': forward_eta_dict.get(taskID, 'None'),
                    'backward_eta': backward_eta_dict.get(taskID, 'None'),
                } for taskID in loss_dict
            }
            # Remove forward_etas, backward_etas, FTs, BTs, loss, length
            for key in ['forward_etas', 'backward_etas', 'FTs', 'BTs', 'loss', 'length']:
                self.record_dict.pop(key)
            
            # os.makedirs(self.profile_file, exist_ok=True)
            with open(self.profile_file, 'w') as f:
                json.dump(self.record_dict, f, indent=4)
            
        else:
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
        active_selection = f"active_{self.active_selection}" if self.active_selection is not None else "active_1.0"
        # task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta}|tau={self.epsilon})" if self.task_assignment == 'workload' else self.task_assignment
        task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta}|tau={self.epsilon})" if (self.task_assignment == 'workload' or '+' in self.task_assignment) else self.task_assignment
        for nodeID, timing_info in self.timing_infos.items():
            if self.no_prior_profile:
                stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-prior-profile_node{nodeID}.json'
            elif self.no_memory_check:
                stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-memory-check_node{nodeID}.json'
            elif self.no_prioritization:
                stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-prioritization_node{nodeID}.json'
            else:
                stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_node{nodeID}.json'
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
        total_runtime = 0
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
            if node_min_time == float('inf') or node_max_time == float('-inf'):
                continue
            else:
                total_runtime += node_max_time - node_min_time
                    
        bubble_rate = sum(total_idles) / sum(total_latencies) if sum(total_latencies) > 0 else 0
        weight_sync_times = None
        for key, value in metrics.items():
            if key == 'train_loss':
                train_losses = value
            elif key == 'inference_loss':
                inference_losses = value
            elif key == 'weight_sync':
                weight_sync_times = value
            metrics[key] = sum(value) / len(value) if value else 0
        
        # Calculate response times
        metrics['num_tasks'] = self.total_tasks
        metrics['retrain_tasks'] = self.retraining_tasks
        metrics['actual_retrained_tasks'] = len(self._trained_task_lengths)
        metrics['user_tasks'] = len(self.user_task_record)
        metrics['SLO 1x'] = self.slo_threshold
        metrics['bubble_rate'] = bubble_rate 
        metrics['total_runtime'] = total_runtime
        metrics['end2end_latency'] = global_max_time - global_min_time
        # metrics['throughput'] = self.total_tasks / (global_max_time - global_min_time)
        metrics['throughput'] = self.total_tasks / total_runtime
        metrics['num_tasks (node)'] = {nodeID: self.distributed_nodes[nodeID].num_tasks for nodeID in self.distributed_nodes}
        metrics['retrain_tasks (node)'] = {nodeID: self.distributed_nodes[nodeID].train_tasks for nodeID in self.distributed_nodes}
        metrics['user_tasks (node)'] = {nodeID: self.distributed_nodes[nodeID].test_tasks for nodeID in self.distributed_nodes}
        metrics['bubble_rate (node)'] = {
            nodeID: sum(idles) / sum(latencies) if sum(latencies) > 0 else 0 
            for nodeID, idles, latencies in zip(node_idles.keys(), node_idles.values(), node_latencies.values())
        }
        metrics['end2end_latency (node)'] = {nodeID: node_timelines[nodeID][1] - node_timelines[nodeID][0] for nodeID in node_timelines}
        metrics['throughput (node)'] = {nodeID: self.distributed_nodes[nodeID].num_tasks / (node_timelines[nodeID][1] - node_timelines[nodeID][0]) for nodeID in node_timelines}
        metrics['node_timelines'] = node_timelines
        metrics['idles_sum'] = sum(total_idles)
        metrics['idles_sum (node)'] = {nodeID: sum(idles) for nodeID, idles in node_idles.items()}
        metrics['idles_avg'] = sum(total_idles) / len(total_idles) if total_idles else 0
        metrics['idles_avg (node)'] = {nodeID: sum(idles) / len(idles) if idles else 0 for nodeID, idles in node_idles.items()}
        if weight_sync_times is not None:
            metrics['weight_sync_times'] = weight_sync_times
        metrics['length_statistics'] = {
            'mean': self.mean_length,
            'std': self.std_length,
            'medium': self.medium_length,
            'min': self.min_length,
            'max': self.max_length,
        }
        metrics['length_statistics (node)'] = {}
        for nodeID in self.distributed_nodes:
            node_lengths = [task['length'] for task in self.distributed_nodes[nodeID].task_allocation]
            if not node_lengths:
                continue
            medium_length = np.median(node_lengths)
            min_length = min(node_lengths)
            max_length = max(node_lengths)
            metrics['length_statistics (node)'][nodeID] = {
                'mean': self.distributed_nodes[nodeID].mean,
                'std': self.distributed_nodes[nodeID].std,
                'medium': medium_length,
                'min': min_length,
                'max': max_length,
            } 
        
        if self.user_task_record:
            # total_response_time, total_wait_time, total_inference_time = 0, 0, 0
            response_times, wait_times, latencies = [], [], []
            user_global_min_time, user_global_max_time = float('inf'), float('-inf')
            for taskID, record_dict in self.user_task_record.items():
                # total_response_time += record_dict['end'] - record_dict['release']
                # total_wait_time += record_dict['start'] - record_dict['release']
                # total_inference_time += record_dict['end'] - record_dict['start']
                if 'start' not in record_dict or 'end' not in record_dict:
                    print(f"Unrecorded user request (ID={taskID})!")
                    continue
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
            metrics['SLO attainment (1x)'] = sum([1 for rt in response_times if rt <= self.slo_threshold]) / len(response_times) if response_times else 0
            metrics['SLO attainment (5x)'] = sum([1 for rt in response_times if rt <= 5 * self.slo_threshold]) / len(response_times) if response_times else 0
            metrics['SLO attainment (10x)'] = sum([1 for rt in response_times if rt <= 10 * self.slo_threshold]) / len(response_times) if response_times else 0
            metrics['user_task_record'] = self.user_task_record
            
        metrics['idles'] = total_idles # list
        if self.training_taskIDs:
            metrics['train_losses'] = train_losses # list
        if self.inference_taskIDs:
            metrics['inference_losses'] = inference_losses # list
        # metrics['losses (node)'] = {nodeID: list(self.distributed_nodes[nodeID].losses.values()) for nodeID in self.distributed_nodes} 
        metrics['task_stats (node)'] = {
            nodeID: self.distributed_nodes[nodeID].task_allocation 
            for nodeID in self.distributed_nodes
        }   
        
        # Save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        length_heterogeneity = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
        active_selection = f"active_{self.active_selection}" if self.active_selection is not None else "active_1.0"
        task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta}|tau={self.epsilon})" if (self.task_assignment == 'workload' or '+' in self.task_assignment) else self.task_assignment
        if self.no_prior_profile:
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-prior-profile_ID={self.experimentID}.json'
        elif self.no_memory_check:
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-memory-check_ID={self.experimentID}.json'
        elif self.no_prioritization:
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-prioritization_ID={self.experimentID}.json'
        else:
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_ID={self.experimentID}.json'
        # with open(stats_f, 'w') as f:
        #     json.dump(metrics, f, indent=4)
        # print(f"Metrics saved to {stats_f}")
        save_metrics_with_order(metrics, stats_f)
        

    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, help='model name or path')
    parser.add_argument('--model_name', type=str, default='dummy', help='model name')
    parser.add_argument('--memory_threshold', type=float, default=0.5, 
                        help='threshold for maximum memory allocation in each GPU device')
    parser.add_argument('--max_wait', type=float, default=10, 
                        help='maximum time to wait from available memory')
    parser.add_argument('--access_token', type=str, default=None, help='access token')
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_length', action='store_true', help='save the length of each task')
    parser.add_argument('--setting', type=str, default='active', choices=['active','interval','isolated'], 
                        help='training setting')
    parser.add_argument('--isolated_split', type=float, default=None, 
                        help='split ratio for isolated test & train nodes. If not provided, the retraining rate is used.')
    parser.add_argument('--priority', type=str, default='FIFO', help='scheduling priority, default: FIFO')
    parser.add_argument('--task_assignment', type=str, default='random', choices=['rr', 'random', 'workload', 'util', 'rr+', 'random+', 'util+'], 
                        help='node level scheduling policy')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1, help='proportion of training tasks')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--rate_lambda', type=int, default=10, help='Average number of inference requests per second')
    parser.add_argument('--alpha', type=float, default=1, help='response time coefficient')
    parser.add_argument('--beta', type=float, default=1, help='length heterogeneity coefficient')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='small value to avoid division by zero')
    parser.add_argument('--k', type=float, default=0.5, help='weight for balancing the loss and length consistency for adaptive training')
    parser.add_argument('--workload', type=str, default='poisson', help='workload arrival pattern')
    parser.add_argument('--length_distribution', type=str, default='random', choices=['random', 'ascending', 'descending', 'bursty'], 
                        help='distribution of input sequence length')
    parser.add_argument('--length_heterogeneity', type=int, default=None, 
                        help='standard deviation of the length distribution of the sampled subset')
    parser.add_argument('--active_selection', type=str, default=None,
                        help='active selection ratio for training tasks')
    parser.add_argument('--profile_dir', type=str, default='profile', help='directory to save profiling results')
    parser.add_argument('--output_dir', type=str, default='prof')
    parser.add_argument('--experiments', type=int, default=1, help='number of experiments')
    parser.add_argument('--run_mode', type=str, default='online', choices=['online', 'offline'], help='Whether to use RECORD MODEL for offline profiling')
    parser.add_argument('--PP', type=str, default='async', choices=['sync', 'async'], help='Implement A-PP or S-PP for training tasks')
    parser.add_argument('--no_prior_profile', action='store_true', help='Whether to use offline profiling results as prior')
    parser.add_argument('--no_memory_check', action='store_true', help='Whether to use memory checker before each execution')
    parser.add_argument('--no_prioritization', action='store_true', help='Whether to use task prioritization (for LeMix only)')
    args = parser.parse_args()
    
    for i in range(args.experiments):
        run_experiment(args, i)
    
