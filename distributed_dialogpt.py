import os
import sys
sys.dont_write_bytecode = True
import queue
import argparse
import pdb
from typing import List, Dict, Optional, Any, Union
import torch
import logging
from utils import record_time, Task
from distributed_llm import DistributedLLM
from models import (
    # GPTStartingStage,
    # GPTIntermediateStage,
    # GPTEndingStage,
    CustomizedGPT2Out,
    _prepare_decoding_inputs,
)

# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 

class DistributedDialoGPT(DistributedLLM):
    
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.model_n = args.model_name
        self.ckpt_path = f'{self.output_dir}/stages_{self.model_n}_{self.setting}_{self.workload}_{self.retraining_rate}'

    def device_inference(
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
            priority, taskID = deviceQueue.get()
            # if taskID is None:
            if taskID == float('inf'):
                # Signal that this thread is done
                print("Stage {} finished inference".format(device))
                if nextdeviceQueue is not None: # intermediate stage
                    # nextdeviceQueue.put(None)
                    nextdeviceQueue.put((float('inf'), float('inf')))
                break
            
            task = preloaded_tasks[taskID]
            # assert task.task_id == taskID
            inputs = task.hiddens[stageID]
            
            if inputs is None:
                print("Stage {} waiting for task {}".format(device, taskID))
                continue    
                
            if stageID == 0: # prepare inputs
                inputs = _prepare_decoding_inputs(inputs)
                task.feedback = inputs.pop('labels', None)
                
            tuple_outputs = self.forward(task, inputs, stageID, nodeID, device, timing_info)
            task.hiddens[stageID] = None # clear the input that is no longer needed
                
            if nextdeviceQueue is not None: # intermediate stage
                # Need to send the output to the next stage, except for the last stage
                outputs = CustomizedGPT2Out(
                    hidden_states=tuple_outputs[0].to(device+1),
                    attention_mask=tuple_outputs[1].to(device+1),
                    head_mask=tuple_outputs[2],
                    encoder_hidden_states=tuple_outputs[3],
                    encoder_attention_mask=tuple_outputs[4],
                    all_hidden_states=tuple_outputs[5],
                    all_self_attentions=tuple_outputs[6],
                    all_cross_attentions=tuple_outputs[7],
                    output_shape=tuple_outputs[8],
                )   
                task.hiddens[stageID+1] = outputs
                nextdeviceQueue.put((priority, taskID))
                
            else: # last stage
                # outputs = CausalLMOutputWithCrossAttentions(
                #     loss=tuple_outputs[0],
                #     logits=tuple_outputs[1],
                #     past_key_values=tuple_outputs[2],
                #     hidden_states=tuple_outputs[3],
                #     attentions=tuple_outputs[4],
                #     cross_attentions=tuple_outputs[5],
                # )
                # loss = outputs.loss
                loss = tuple_outputs[0]
                # print("[NLL loss={}] stage {} finished task {}".format(loss, device, taskID))
                self.metrics["loss"].append(loss.item())
                
                if task.require_training:
                    # Backprop on the last stage
                    try:
                        loss.backward()
                        record_time(init_device, 'end', 'backward', timing_info)
                    except Exception as e:
                        # logging.error(f"[node {nodeID} | stage {stageID}] Backward error occurred: {e}")
                        pass
                    self._trained_tasks += 1
                    
                    # Optimization
                    try:
                        self.distributed_optimizers[nodeID].step()
                        self.distributed_schedulers[nodeID].step()
                        self.distributed_optimizers[nodeID].zero_grad() # clear gradients
                    except Exception as e:
                        # logging.error(f"[node {nodeID} | stage {stageID}] Optimization error occurred: {e}")
                        pass
                    print("Stage {} finish backward propagation for task {} !".format(device, taskID))
                    
                    if (self.setting == 'isolated') and (self._trained_tasks % self.saving_steps == 0): 
                        # Save the parameters of stages in the last node and load them in other nodes
                        print(f" *** Save checkpoint {self.ckpt_path} *** ")
                        for j in range(self.num_gpus_per_node):
                            torch.save(self.distributed_stages[nodeID][j].state_dict(), f"{self.ckpt_path}_stage{j}.pt")
                        # For other nodes, load the parameters from the last node
                        for i in range(self.num_nodes - 1):
                            print(f" *** Load checkpoint for Node {i} *** ")
                            for j in range(self.num_gpus_per_node):
                                self.distributed_stages[i][j].load_state_dict(torch.load(f"{self.ckpt_path}_stage{j}.pt"))
                                
                # else:
                #     task.hiddens.append(loss)
                #     deviceQueue.put(taskID) # put it back to the queue

    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/DialoGPT-small', help='model name or path')
    parser.add_argument('--model_name', type=str, default='dialogpt', help='model name')
    parser.add_argument('--access_token', type=str, default=None, help='access token')
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--setting', type=str, default='active', choices=['active','interval', 'isolated'], help='training setting')
    parser.add_argument('--priority', type=str, default=None, help='scheduling priority')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--workload', type=str, default='poisson', choices=['poisson', 'all'], help='workload arrival pattern')
    parser.add_argument('--output_dir', type=str, default='prof')
    args = parser.parse_args()
    
    distributed_llm = DistributedDialoGPT(args)
    distributed_llm.run()
