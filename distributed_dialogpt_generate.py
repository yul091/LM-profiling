import os
import sys
sys.dont_write_bytecode = True
import queue
import random
import pdb
import argparse
import time
from typing import List, Dict, Optional, Any, Union
import torch
from torch.nn.utils.rnn import pad_sequence
import logging
from utils import record_time, Task
from distributed_llm_generate import DistributedLLM, set_seed, gc
from models import CustomizedGPT2Out
from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria, MinLengthLogitsProcessor
# from memory_profiler import memory_usage
# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Custom padding function
def pad_batch(tensor_list, pad_value=0):
    """
    Pads a list of tensors to the same length along dimension 1.
    """
    return pad_sequence(tensor_list, batch_first=True, padding_value=pad_value)



def run_experiment(args: argparse.Namespace, experimentID: int):
    print(f"\n ** Experiment {experimentID+1} **\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize and run the distributed model
    distributed_llm = DistributedDialoGPT(args, experimentID=experimentID)
    distributed_llm.run()
    record_mode = distributed_llm.RECORD_MODE
    
    # Clean up resources explicitly
    del distributed_llm
    torch.cuda.empty_cache()
    gc.collect()

    # Rerun if necessary based on specific conditions
    if record_mode and args.run_mode == 'online':  # Assuming record_mode is a valid arg
        new_run = DistributedDialoGPT(args, experimentID=experimentID)
        new_run.run()
        
        # Final clean up
        del new_run
        torch.cuda.empty_cache()
        gc.collect()
 

class DistributedDialoGPT(DistributedLLM):
    
    def __init__(self, args: argparse.Namespace, experimentID: int = 0):
        super().__init__(args, experimentID)

    def stage_inference(
        self, 
        stageID: int,
        nodeID: int,
        timing_info: Dict[str, List[float]], 
        preloaded_tasks: List[Task], 
        deviceQueue: Union[queue.Queue, queue.PriorityQueue],
        max_length: int = 128,
        max_wait_time: float = 0.1, 
        nextdeviceQueue: Optional[Union[queue.Queue, queue.PriorityQueue]] = None,
        init_device: Optional[int] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ):
        init_device = init_device if init_device is not None else self.distributed_nodes[nodeID].init_device
        device = init_device + stageID
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList([MinLengthLogitsProcessor(10, eos_token_id=self.tokenizer.eos_token_id),])
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
        
        while True:
            priority, taskID = deviceQueue.get()
            if taskID == float('inf'):
                # Signal that this thread is done
                print("Stage {} finished inference".format(device))
                if nextdeviceQueue is not None: # intermediate stage
                    # nextdeviceQueue.put(None)
                    nextdeviceQueue.put((float('inf'), float('inf')))
                break
            
            task: Task = preloaded_tasks[taskID]  # task.query: ['input_ids', 'attention_mask', 'labels']
            assert task.task_id == taskID
    
            # For inference tasks, handle batching dynamically at the first stage
            if stageID == 0 and not task.require_training:
                while True:
                    if not self.node_batch_executing[nodeID]:  # If the current batch is done, exit the loop
                        break
                    time.sleep(0.01)

                if not self.node_batches[nodeID]:
                    task.start = time.time()
                
                if len(self.node_batches[nodeID]) < self.batch_size: 
                    self.node_batches[nodeID].append(task)
                else:  # if the batch is full, put the new task back and execute the batch
                    deviceQueue.put((priority, taskID))

                # Append the remaining tasks (in the queue) to the current batch if it doesn't exceed batch size or max wait time
                while len(self.node_batches[nodeID]) < self.batch_size:
                    # print(f"Current batch size: {len(self.node_batches[nodeID])}, remaining tasks: {deviceQueue.queue}")
                    try:
                        tp, td = deviceQueue.get(timeout=max_wait_time)
                        if td == float('inf'):
                            deviceQueue.put((tp, td))
                            break
                        tt: Task = preloaded_tasks[td]
                        if tt.require_training:
                            deviceQueue.put((tp, td))
                            break
                        self.node_batches[nodeID].append(tt)
                    except queue.Empty:
                        break

                # Use the data collator to pad and combine the batch
                input_ids_list = [t.query['input_ids'].squeeze(0) for t in self.node_batches[nodeID]]
                attention_mask_list = [t.query['attention_mask'].squeeze(0) for t in self.node_batches[nodeID]]
                input_ids = pad_batch(input_ids_list, pad_value=self.tokenizer.pad_token_id)
                attention_mask = pad_batch(attention_mask_list, pad_value=0)  

                # Update the current task to use the collated batch
                task.hiddens[stageID] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                }
                self.node_batch_executing[nodeID] = True

            inputs = task.hiddens[stageID]

            if inputs is None:
                print("Stage {} waiting for task {}".format(device, taskID))
                continue  

            batch_size, input_length = task.hiddens[0]['input_ids'].shape
            if stageID == 0: # prepare inputs
                task.feedback = inputs.pop('labels', None)
            
            # Memory check and scale down if necessary
            if self.wait_for_device_availability(device):
                tuple_outputs = self.forward(
                    taskID, inputs, input_length, stageID, nodeID, device, timing_info, 
                    require_training=task.require_training, 
                    labels=task.feedback,
                )
            else:
                tuple_outputs = None
                print(f"Failed waiting for device {device} to be available, dropping task {taskID}")
            # tuple_outputs = self.forward(task, inputs, stageID, nodeID, device, timing_info)
            if stageID > 0:
                task.hiddens[stageID] = None # clear the input that is no longer needed
             
            if tuple_outputs is None:  # Error occurred
                continue
                
            if nextdeviceQueue is not None: # intermediate stage
                # Need to send the output to the next stage, except for the last stage
                task.hiddens[stageID+1] = CustomizedGPT2Out(
                    hidden_states=tuple_outputs[0].to(device+1),
                    attention_mask=tuple_outputs[1].to(device+1) if tuple_outputs[1] is not None else None,
                    head_mask=tuple_outputs[2],
                    encoder_hidden_states=tuple_outputs[3],
                    encoder_attention_mask=tuple_outputs[4],
                    all_hidden_states=tuple_outputs[5],
                    all_self_attentions=tuple_outputs[6],
                    all_cross_attentions=tuple_outputs[7],
                    output_shape=tuple_outputs[8],
                )   
                nextdeviceQueue.put((priority, taskID))
                
            else: # last stage
                loss = tuple_outputs[0]
                if task.require_training:
                    self.metrics["train_loss"].append(loss.item())
                    print("[Train loss={}] stage {} finished task {}".format(loss, device, taskID))
                else:
                    self.metrics["inference_loss"].append(loss.item())
                    print("[Inference loss={}] stage {} finished task {}".format(loss, device, taskID))
                    # Finished sentences should have their next token be a padding token
                    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=init_device)
                    next_token_logits = tuple_outputs[1][:, -1, :].to(init_device)  # B X V
                    # pre-process distribution
                    next_tokens_scores = logits_processor(task.hiddens[0]["input_ids"], next_token_logits)
                    next_tokens = torch.argmax(next_tokens_scores, dim=-1)  # Greedy decoding (B)
                    # next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                    # Update unfinished sequences
                    unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(self.tokenizer.eos_token_id).long())
                    # print(f"Unfinished sequences: {unfinished_sequences}")
                    # Stop when each sentence is finished or if we exceed the maximum length
                    if unfinished_sequences.max() == 0 or stopping_criteria(task.hiddens[0]["input_ids"], next_tokens_scores):
                        self.node_batches[nodeID].clear()  # Clear the batch
                    else:
                        finished_indices = (unfinished_sequences == 0).nonzero(as_tuple=True)[0]
                        # Append generated tokens to the sequence for all tasks in the batch
                        for i, t in enumerate(self.node_batches[nodeID]):
                            t.query['input_ids'] = torch.cat([t.query['input_ids'], next_tokens[i].view(1, 1)], dim=1)
                            # print(f"\tUpdated input_ids: {t.query['input_ids'].shape}")
                            t.query['attention_mask'] = torch.cat(
                                [t.query['attention_mask'], torch.ones((1, 1), device=init_device)], dim=1
                            )
                            print(f"[Current batch  {len(self.node_batches[nodeID])} | finished {finished_indices}] task {t.task_id} (index {i}) -> next token {next_tokens[i]}")
                        # Remove finished sequences from the batch
                        if len(finished_indices) > 0:
                            print(f"Removing finished sequences: {finished_indices}")
                            self.node_batches[nodeID] = [
                                t for i, t in enumerate(self.node_batches[nodeID]) if i not in finished_indices
                            ]
                        print(f"Remaining tasks: {len(self.node_batches[nodeID])}")
                    self.node_batch_executing[nodeID] = False
                    task.hiddens[0] = None  # Clear the input that is no longer needed
                
                if self.RECORD_MODE:
                    self.record_dict['loss'].append((loss.item(), taskID))
                    self.record_dict['length'].append((input_length, taskID))
                
                if task.require_training and self.wait_for_device_availability(device):
                    bb = time.time()
                    self.task_trace[nodeID][self.num_gpus_per_node-1]['bb'] = bb
                    if taskID in self.train_trace[nodeID]:
                        self.train_trace[nodeID][taskID][self.num_gpus_per_node-1]['bb'] = bb # update the recorded time
                        self.all_trace[nodeID][taskID][self.num_gpus_per_node-1]['bb'] = bb # update the recorded time

                    # Do backward propagation
                    try:
                        loss.backward()
                        be = record_time(init_device, 'end', 'backward', taskID, timing_info)
                        self.task_trace[nodeID][0]['be'] = be
                        if taskID in self.train_trace[nodeID]:
                            self.train_trace[nodeID][taskID][0]['be'] = be   
                            self.all_trace[nodeID][taskID][0]['be'] = be
                        
                        if self.backward_eta is None:
                            self.backward_eta = (be - bb) / (self.num_gpus_per_node * batch_size * input_length ** 2)
                        print("Stage {} finish backward propagation for task {} !".format(device, taskID))
                        
                    except Exception as e:
                        # logging.error(f"[node {nodeID} | stage {stageID}] Backward error occurred: {e}")
                        pass

                    if self.PP == 'sync':  # If S-PP is used, we need to let the scheduler know that the task is done
                        self.training_status[taskID] = True
                    
                    self._trained_task_lengths.append(input_length)

                    if self.RECORD_MODE:  # In RECORD_MODE, we do not update the model parameters
                        self.record_dict['backward_etas'].append(
                            ((be - bb) / (self.num_gpus_per_node * batch_size * input_length ** 2), taskID)
                        )
                        self.record_dict['BTs'].append(((be - bb)/ self.num_gpus_per_node, taskID))
                        
                    else: # Optimization
                        try:
                            self.distributed_optimizers[nodeID].step()
                            self.distributed_schedulers[nodeID].step()
                        except Exception as e:
                            logging.error(f"[node {nodeID} | stage {stageID}] Optimization error occurred: {e}")
                            # pass
                    
                    self.distributed_optimizers[nodeID].zero_grad() # clear gradients
                    
                    if (self.setting == 'isolated') and (len(self._trained_task_lengths) % self.saving_steps == 0): 
                        # Save the parameters of stages in the last node and load them in other nodes
                        print(f" *** Save checkpoint {self.ckpt_path} *** ")
                        sync_start = time.time()
                        for j in range(self.num_gpus_per_node):
                            torch.save(self.distributed_stages[nodeID][j].state_dict(), f"{self.ckpt_path}_stage{j}.pt")
                        # For other nodes, load the parameters from the last node
                        # for i in range(self.num_nodes - 1):
                        for i in self._test_nodes:
                            print(f" *** Load checkpoint for Node {i} *** ")
                            for j in range(self.num_gpus_per_node):
                                self.distributed_stages[i][j].load_state_dict(torch.load(f"{self.ckpt_path}_stage{j}.pt"))
                        sync_end = time.time()
                        self.metrics['weight_sync'].append(sync_end - sync_start)
                    
                    self._training_step += 1   
                
                # Update task stats
                if task.require_training:
                    self.distributed_nodes[nodeID].updata_task_stats(
                        taskID=taskID, 
                        loss=loss.item(), 
                        length=input_length, 
                        computation_type='training',
                    )         
                else:  # for each task in the batch, we need to update the stats
                    for t in self.node_batches[nodeID]:
                        self.distributed_nodes[nodeID].updata_task_stats(
                            taskID=t.task_id, 
                            loss=loss.item(), 
                            length=t.query['input_ids'].shape[1], 
                            computation_type='test',
                        )    
                

    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, help='model name', default='microsoft/DialoGPT-large')
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
    parser.add_argument('--alpha', type=float, default=1, help='response time coefficient')
    parser.add_argument('--beta', type=float, default=1, help='length heterogeneity coefficient')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='small value to avoid division by zero')
    parser.add_argument('--rate_lambda', type=int, default=10, help='Average number of inference requests per second')
    parser.add_argument('--workload', type=str, default='poisson', help='workload arrival pattern')
    parser.add_argument('--length_distribution', type=str, default='random',
                        help='distribution of input sequence length')
    parser.add_argument('--length_heterogeneity', type=int, default=None, 
                        help='standard deviation of the length distribution of the sampled subset')
    parser.add_argument('--active_selection', type=str, default=None,
                        help='active selection ratio for training tasks')
    parser.add_argument('--k', type=float, default=0.5, help='weight for balancing the loss and length consistency for adaptive training')
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
        
    
    # for i in range(args.experiments):
    #     print(f"\n ** Experiment {i+1} **\n")
    #     distributed_llm = DistributedDialoGPT(args, experimentID=i)
    #     distributed_llm.run()
        
    #     # If run_mode is online and RECORD_MODE is activated, we need to re-run the experiment
    #     if distributed_llm.RECORD_MODE and args.run_mode == 'online':
    #         del distributed_llm
    #         torch.cuda.empty_cache()
    #         new_run =  DistributedDialoGPT(args, experimentID=i)
    #         new_run.run()

