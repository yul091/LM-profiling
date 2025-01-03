import os
import sys
sys.dont_write_bytecode = True
import queue
import argparse
import time
from typing import List, Dict, Optional, Any, Union
import logging
from torch.cuda.amp import autocast
from utils import record_time, Task
from distributed_llm import DistributedLLM, set_seed, gc
from models import (
    CustomizedLlamaOut,
    _prepare_inputs,
)
import torch
from torch.cuda.amp import GradScaler
scaler = GradScaler()

# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run_experiment(args: argparse.Namespace, experimentID: int):
    print(f"\n ** Experiment {experimentID+1} **\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize and run the distributed model
    distributed_llm = DistributedLlama(args, experimentID=experimentID)
    distributed_llm.run()
    record_mode = distributed_llm.RECORD_MODE
    
    # Clean up resources explicitly
    del distributed_llm
    torch.cuda.empty_cache()
    gc.collect()

    # Rerun if necessary based on specific conditions
    if record_mode and args.run_mode == 'online':  # Assuming record_mode is a valid arg
        new_run = DistributedLlama(args, experimentID=experimentID)
        new_run.run()
        
        # Final clean up
        del new_run
        torch.cuda.empty_cache()
        gc.collect()
        
        
class DistributedLlama(DistributedLLM):
    
    def __init__(self, args: argparse.Namespace, experimentID: int = 0):
        super().__init__(args, experimentID)

    def stage_inference(
        self, 
        stageID: int,
        nodeID: int,
        timing_info: Dict[str, List[float]], 
        preloaded_tasks: List[Task], 
        deviceQueue: Union[queue.Queue, queue.PriorityQueue],
        nextdeviceQueue: Optional[Union[queue.Queue, queue.PriorityQueue]] = None,
        init_device: Optional[int] = None,
    ):
        device = self.distributed_nodes[nodeID].init_device + stageID
        init_device = init_device if init_device is not None else self.distributed_nodes[nodeID].init_device
        
        while True:
            # log_queue_contents(deviceQueue, nodeID, stageID)
            priority, taskID = deviceQueue.get()
            # print(f"[Node {nodeID} | Stage {stageID}] Retrieved task {taskID} from deviceQueue")

            if taskID == float('inf'):
                # Signal that this thread is done
                print(f"[Node {nodeID} | Stage {stageID}] Received termination signal, ending inference.")
                if nextdeviceQueue is not None:  # Propagate termination signal to the next stage
                    nextdeviceQueue.put((float('inf'), float('inf')))
                break
            
            task: Task = preloaded_tasks[taskID]
            batch_size, input_length = task.query['input_ids'].shape
            assert task.task_id == taskID
            inputs = task.hiddens[stageID]
            
            if inputs is None:
                print(f"[Node {nodeID} | Stage {stageID}] Waiting for inputs for task {taskID}")
                continue    
                
            if stageID == 0: # prepare inputs
                task.feedback = inputs.pop('labels', None)

            # Memory check and scale down if necessary
            if self.wait_for_device_availability(device):
                tuple_outputs = self.forward(task, inputs, stageID, nodeID, device, timing_info)
            else:
                tuple_outputs = None
                print(f"Failed waiting for device {device} to be available, dropping task {taskID}")
            # tuple_outputs = self.forward(task, inputs, stageID, nodeID, device, timing_info)
            task.hiddens[stageID] = None # clear the input that is no longer needed

            if tuple_outputs is None:  # Error occurred
                continue
                
            if nextdeviceQueue is not None:  # Intermediate stage
                # Need to send the output to the next stage, except for the last stage
                task.hiddens[stageID+1] = CustomizedLlamaOut(
                    hidden_states=tuple_outputs[0].to(device+1),
                    past_key_values=_prepare_inputs(tuple_outputs[1], device+1),
                    all_hidden_states=tuple_outputs[2],
                    all_self_attns=tuple_outputs[3],
                    position_ids=tuple_outputs[4].to(device+1) if tuple_outputs[4] is not None else None,
                    attention_mask=tuple_outputs[5].to(device+1) if tuple_outputs[5] is not None else None,
                )
                nextdeviceQueue.put((priority, taskID))
                # print(f"[Node {nodeID} | Stage {stageID}] Passed task {taskID} to next stage")
                
            else: # last stage
                loss = tuple_outputs[0]
                # self.metrics["loss"].append(loss.item())
                if task.require_training:
                    self.metrics["train_loss"].append(loss.item())
                else:
                    self.metrics["inference_loss"].append(loss.item())
                # print(f"[Node {nodeID} | Stage {stageID}] Finished task {taskID} with loss {loss} (tuple outputs {tuple_outputs})")

                if self.RECORD_MODE:
                    self.record_dict['loss'].append((loss.item(), taskID))
                    self.record_dict['length'].append((input_length, taskID))
                
                if task.do_backward and self.wait_for_device_availability(device, force_check=False):
                    # Backprop on the last stage
                    bb = time.time()
                    self.task_trace[nodeID][self.num_gpus_per_node-1]['bb'] = bb
                    if taskID in self.train_trace[nodeID]:
                        self.train_trace[nodeID][taskID][self.num_gpus_per_node-1]['bb'] = bb # update the recorded time
                        self.all_trace[nodeID][taskID][self.num_gpus_per_node-1]['bb'] = bb # update the recorded time
                    try:
                        # loss.backward()
                        scaler.scale(loss).backward()
                        be = record_time(init_device, 'end', 'backward', taskID, timing_info)
                        self.task_trace[nodeID][0]['be'] = be
                        if taskID in self.train_trace[nodeID]:
                            self.train_trace[nodeID][taskID][0]['be'] = be   
                            self.all_trace[nodeID][taskID][0]['be'] = be
                        
                        if self.backward_eta is None:
                            self.backward_eta = (be - bb) / (self.num_gpus_per_node * batch_size * input_length ** 2)
                            
                        print("Stage {} finish backward propagation for task {} !".format(device, taskID))
                    except Exception as e:
                        # logging.error(f"[node {nodeID} | stage {stageID} | task {taskID}] Backward error occurred: {e}")
                        pass

                    if self.PP == 'sync':  # If S-PP is used, we need to let the scheduler know that the task is done
                        self.training_status[taskID] = True
                    
                    self._trained_task_lengths.append(input_length)

                    if self.RECORD_MODE:  # In RECORD_MODE, we do not update the model parameters
                        self.record_dict['backward_etas'].append(
                            ((be - bb) / (self.num_gpus_per_node * batch_size * input_length ** 2), taskID)
                        )
                        self.record_dict['BTs'].append(((be - bb)/self.num_gpus_per_node, taskID))
                        
                    else:  # Optimization
                        try:
                            # self.distributed_optimizers[nodeID].step()
                            scaler.step(self.distributed_optimizers[nodeID])
                            self.distributed_schedulers[nodeID].step()
                            scaler.update()
                        except Exception as e:
                            # logging.error(f"[node {nodeID} | stage {stageID} | task {taskID}] Optimization error occurred: {e}")  
                            pass     
                    
                    self.distributed_optimizers[nodeID].zero_grad() # clear gradients
                    
                    if (self.setting == 'isolated') and (len(self._trained_task_lengths) % self.saving_steps == 0): 
                        # Save the parameters of stages in the last node and load them in other nodes
                        print(f" *** Save checkpoint {self.ckpt_path} *** ")
                        for j in range(self.num_gpus_per_node):
                            torch.save(self.distributed_stages[nodeID][j].state_dict(), f"{self.ckpt_path}_stage{j}.pt")
                        # For other nodes, load the parameters from the last node
                        for i in self._test_nodes:
                            print(f" *** Load checkpoint for Node {i} *** ")
                            for j in range(self.num_gpus_per_node):
                                self.distributed_stages[i][j].load_state_dict(torch.load(f"{self.ckpt_path}_stage{j}.pt"))
                    
                    self._training_step += 1

                # Update task stats
                self.distributed_nodes[nodeID].updata_task_stats(
                    taskID=taskID, 
                    loss=loss.item(), 
                    length=input_length, 
                    computation_type='training' if task.do_backward else 'test',
                )             

    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama//Llama-2-7b-chat-hf', help='model name or path')
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--access_token', type=str, default=None, help='access token')
    parser.add_argument('--memory_threshold', type=float, default=0.8, help='threshold for maximum memory allocation in each GPU device')
    parser.add_argument('--max_wait', type=float, default=10, 
                        help='maximum time to wait from available memory')
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_length', action='store_true', help='save the length of each task')
    parser.add_argument('--setting', type=str, default='active', choices=['active', 'interval', 'isolated'], 
                        help='training setting')
    parser.add_argument('--isolated_split', type=float, default=None, 
                        help='split ratio for isolated test and train nodes')
    parser.add_argument('--priority', type=str, default='FIFO', choices=['FIFO', 'MLF', 'LLF'], help='scheduling priority')
    parser.add_argument('--task_assignment', type=str, default='random', choices=['rr', 'random', 'workload', 'util', 'rr+', 'random+', 'util+'], 
                        help='node level scheduling policy')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1, help='response time coefficient')
    parser.add_argument('--beta', type=float, default=1, help='length heterogeneity coefficient')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='small value to avoid division by zero')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--rate_lambda', type=int, default=10, help='Average number of tasks produced per second')
    parser.add_argument('--workload', type=str, default='poisson', choices=['poisson', 'all'], help='workload arrival pattern')
    parser.add_argument('--length_distribution', type=str, default='random', choices=['random', 'ascending', 'descending', 'bursty'], 
                        help='distribution of input sequence length')
    parser.add_argument('--length_heterogeneity', type=int, default=None, 
                        help='standard deviation of the length distribution of the sampled subset')
    parser.add_argument('--active_selection', type=float, default=None,
                         help='active selection ratio for training tasks')
    parser.add_argument('--k', type=float, default=0.5, help='weight for balancing the loss and length consistency for adaptive training')
    parser.add_argument('--output_dir', type=str, default='prof')
    parser.add_argument('--profile_dir', type=str, default='profile', help='directory to save profiling results')
    parser.add_argument('--experiments', type=int, default=1, help='number of experiments')
    parser.add_argument('--run_mode', type=str, default='online', choices=['online', 'offline'], help='Whether to use RECORD MODEL for offline profiling')
    parser.add_argument('--PP', type=str, default='async', choices=['sync', 'async'], help='Implement A-PP or S-PP for training tasks')
    parser.add_argument('--no_prior_profile', action='store_true', help='Whether to use offline profiling results as prior')
    parser.add_argument('--no_memory_check', action='store_true', help='Whether to use memory checker before each execution')
    parser.add_argument('--no_prioritization', action='store_true', help='Whether to use task prioritization (for LeMix only)')
    args = parser.parse_args()
    
    for i in range(args.experiments):
        run_experiment(args, i)

    
