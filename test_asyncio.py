import os
import sys
sys.dont_write_bytecode = True
import time
import json
import queue
import random
import torch
from torch import Tensor
# import asyncio
from tqdm import tqdm
from typing import List, Dict
from torch.nn import TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from dataset import get_data, SentencePairDataset
from models import Encoder, Decoder, PipelineStage
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Subset
from utils import record_time, get_total_params
from collections import defaultdict


class Node:
    def __init__(self, node_id: int, num_gpus_per_node: int, init_device: int = 0):
        self.node_id = node_id
        self.num_gpus_per_node = num_gpus_per_node
        self.device_queues = [queue.Queue() for _ in range(num_gpus_per_node)]
        self.init_device = init_device
        self.last_device = init_device + num_gpus_per_node - 1
        
        
class Task:
    def __init__(self, task_id: int, query: Tensor, feedback: Tensor = None, node_id: int = 0, num_gpus_per_node: int = 1):
        self.task_id = task_id
        self.query = query
        self.hiddens = [query] + [None for _ in range(num_gpus_per_node - 1)]
        self.feedback = feedback
        self.node_id = node_id
        
        
def get_preloaded_dataset(
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
                    task = Task(
                        task_id=i,
                        query=batch[0].cuda(node.init_device), 
                        feedback=batch[1].cuda(node.last_device), 
                        node_id=nodeID,
                        num_gpus_per_node=node.num_gpus_per_node,
                    )
                else:
                    task = Task(
                        task_id=i,
                        query=batch[0].cuda(node.init_device), 
                        node_id=nodeID,
                        num_gpus_per_node=node.num_gpus_per_node,
                    )
                preloaded_tasks[node.node_id].append(task)
                
        elif setting == 'variant':
            for i, batch in enumerate(dataloader):
                # Odd batches are short, better utilize the bubble for retraining
                if i % 2 == 0 and random.random() < 2 * retraining_rate:
                    task = Task(
                        task_id=i,
                        query=batch[0].cuda(node.init_device), 
                        feedback=batch[1].cuda(node.last_device), 
                        node_id=nodeID, 
                        num_gpus_per_node=node.num_gpus_per_node,
                    )
                else:
                    task = Task(
                        task_id=i,
                        query=batch[0].cuda(node.init_device), 
                        node_id=nodeID,
                        num_gpus_per_node=node.num_gpus_per_node,
                    )
                preloaded_tasks[node.node_id].append(task)
    return preloaded_tasks


def producer(taskQueue: queue.Queue, dataloader: DataLoader, rate_lambda: float, workload: str = 'poisson'):
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
    

def globalScheduler(taskQueue: queue.Queue, distributed_nodes: List[Node]):
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


def get_stages(
    ntokens: int, 
    nlayers: int, 
    num_gpus: int, 
    emsize: int, 
    nhead: int, 
    nhid: int, 
    dropout: float, 
    init_device: int = 0,
):
    # Create pipeline stages
    partition_len = ((nlayers - 1) // num_gpus) + 1
    # Add encoder in the beginning.
    tmp_list = [Encoder(ntokens, emsize, dropout)]
    stages = []
    # Add all the necessary transformer blocks
    for i in range(nlayers):
        transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        if i != 0 and i % (partition_len) == 0:
            # Create a new pipeline stage
            stage_device = i // (partition_len) - 1 + init_device
            print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device))
            stages.append(PipelineStage(tmp_list, stage_device))
            tmp_list = []
        tmp_list.append(transformer_block)
        
    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize))
    stages.append(PipelineStage(tmp_list, stage_device + 1))
    print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device + 1))
    print ('Total parameters in model: {:,}'.format(get_total_params(torch.nn.Sequential(*stages))))
    return stages


def device_inference(
    stage: PipelineStage, 
    stageID: int,
    timing_info: dict, 
    preloaded_tasks: List[Task], 
    deviceQueue: queue.Queue,
    nextdeviceQueue: queue.Queue = None,
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
            
        record_time(device, 'start', 'forward', timing_info)
        if task.feedback is not None:
            # This is a retraining task
            output = stage(hidden)
        else:
            with torch.no_grad():
                output = stage(hidden)
        record_time(device, 'end', 'forward', timing_info)
        if nextdeviceQueue is not None:
            # Need to send the output to the next stage, except for the last stage
            task.hiddens[stageID+1] = output.cuda(device+1, non_blocking=True)
            nextdeviceQueue.put(taskID)


def node_inference(
    node: Node,
    preloaded_tasks: List[Task],
    stages: List[PipelineStage], 
    timing_info: dict,
):
    # We use 16 workers to simulateously get task from the queue and inference
    with ThreadPoolExecutor(max_workers=len(stages)) as executor:
        for stageID, stage in enumerate(stages):
            future = executor.submit(
                device_inference, 
                stage, 
                stageID,
                timing_info, 
                preloaded_tasks,
                node.device_queues[stageID],
                node.device_queues[stageID+1] if stageID != len(stages) - 1 else None,
            )
            
    print("Node {} finished inference".format(node.node_id))


def run_stages_concurrently(
    preloaded_tasks: Dict[int, List[Task]],
    distributed_stages: List[List[PipelineStage]], 
    timing_infos: List[dict], 
    distributed_nodes: List[Node],
):
    with ThreadPoolExecutor(max_workers=len(distributed_nodes)) as executor:
        for nodeID, node in enumerate(distributed_nodes):
            future = executor.submit(
                node_inference, 
                node, 
                preloaded_tasks[nodeID], 
                distributed_stages[nodeID], 
                timing_infos[nodeID],
            )
        
        
def main():
    n_samples = args.n_samples
    setting = args.setting
    num_nodes = args.num_nodes
    nlayers = args.nlayers
    emsize = args.emsize
    nhead = args.nhead
    nhid = args.nhid
    dropout = args.dropout
    batch_size = args.batch_size
    block_size = args.block_size
    rate_lambda = args.rate_lambda
    output_dir = args.output_dir
    workload = args.workload
    retraining_rate = args.retraining_rate
    num_gpus_per_node = torch.cuda.device_count() // num_nodes
    distributed_nodes = [
        Node(i, num_gpus_per_node, i * num_gpus_per_node) 
        for i in range(num_nodes)
    ]
    
    # Example data for each stage
    _, _, test_data, vocab = get_data(block_size=block_size, setting=setting)
    dataset = SentencePairDataset(test_data, setting=setting)
    ntokens = len(vocab) # the size of vocabulary
    model_kwargs = {
        'nlayers': nlayers,
        'emsize': emsize,
        'nhead': nhead,
        'nhid': nhid,
        'dropout': dropout,
        'ntokens': ntokens,
    }

    if n_samples > 0:
        if setting == 'random':
            indices = random.sample(range(len(dataset)), n_samples)
        elif setting == 'variant':
            indices = list(range(n_samples))
        dataset = Subset(
            dataset, 
            indices,
        )
        
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

    # Preloaded dataset
    preloaded_tasks = get_preloaded_dataset(distributed_nodes, setting, dataloader, retraining_rate=retraining_rate)

    # Instantiate stages and put them on the correct devices
    distributed_stages = [
        get_stages(
            num_gpus=num_gpus_per_node,
            init_device=distributed_nodes[nodeID].init_device,
            **model_kwargs,
        )
        for nodeID in range(num_nodes)
    ]

    # Run the stages concurrently
    timing_infos = [defaultdict(list) for _ in range(num_nodes)]
    task_queue = queue.Queue()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future1 = executor.submit(
            producer,
            task_queue, 
            dataloader, 
            rate_lambda, 
            workload,
        )
        future2 = executor.submit(
            globalScheduler,
            task_queue,
            distributed_nodes,
        )
        future3 = executor.submit(
            run_stages_concurrently,  
            preloaded_tasks, 
            distributed_stages,
            timing_infos,
            distributed_nodes,
        )

    os.makedirs(output_dir, exist_ok=True)
    for nodeID, timing_info in enumerate(timing_infos):
        # Remove the first start and end time for each GPU
        gpus = list(set(int(key.split('_')[0]) for key in timing_info))
        for gpu_id in gpus:
            timing_info[f'{gpu_id}_start'] = timing_info[f'{gpu_id}_start']
            timing_info[f'{gpu_id}_end'] = timing_info[f'{gpu_id}_end']
        stats_f = f'{output_dir}/timing_info_coroutine_{setting}_{workload}_{retraining_rate}_node{nodeID}.json'
        # stats_f = f'{output_dir}/test_asyncio.json'
        with open(stats_f, 'w') as f:
            json.dump(timing_info, f, indent=4)
    
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--setting', type=str, default='random', choices=['identical','random', 'variant'], help='workload setting')
    parser.add_argument('--nlayers', type=int, default=24)
    parser.add_argument('--emsize', type=int, default=4096)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nhid', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--retraining_rate', type=float, default=0.1)
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--workload', type=str, default='poisson', choices=['poisson', 'all'])
    parser.add_argument('--output_dir', type=str, default='prof')
    args = parser.parse_args()
    
    main()
