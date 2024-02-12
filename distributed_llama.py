import os
import sys
sys.dont_write_bytecode = True
import queue
import logging
import argparse
import pdb
from typing import List, Dict, Optional, Any, Union
import torch
from utils import record_time, Node, Task
from distributed_llm import DistributedLLM
from models import (
    LlamaStartingStage,
    LlamaIntermediateStage,
    LlamaEndingStage, 
    _prepare_inputs,
    _prepare_decoding_inputs,
    CustomizedLlamaOut,
)

# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    

class DistributedLlama(DistributedLLM):
    model_n: str = 'llama2'
    
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)


    def device_inference(
        self,
        stage: Union[LlamaStartingStage, LlamaIntermediateStage, LlamaEndingStage], 
        stageID: int,
        nodeID: int,
        timing_info: Dict[str, List[float]], 
        preloaded_tasks: List[Task], 
        deviceQueue: queue.Queue,
        nextdeviceQueue: queue.Queue = None,
        init_device: Optional[int] = None,
    ):
        device = stage._device
        init_device = init_device if init_device is not None else 0
        
        while True:
            taskID: int = deviceQueue.get()
            if taskID is None:
                # Signal that this thread is done
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
            
            if stageID == 0: # prepare inputs
                inputs = _prepare_decoding_inputs(inputs)
                task.feedback = inputs.pop('labels', None)
            
            tuple_outputs = self.forward(task, inputs, stage, device, timing_info)
            # Clear the input from task.hiddens that is no longer needed
            task.hiddens[stageID] = None
                
            if nextdeviceQueue is not None: # intermediate stage
                outputs = CustomizedLlamaOut(
                    hidden_states=tuple_outputs[0].to(device+1),
                    past_key_values=_prepare_inputs(tuple_outputs[1], device+1),
                    all_hidden_states=tuple_outputs[2],
                    all_self_attns=tuple_outputs[3],
                    position_ids=tuple_outputs[4].to(device+1),
                    attention_mask=tuple_outputs[5].to(device+1),
                )
                # Need to send the output to the next stage, except for the last stage
                task.hiddens[stageID+1] = outputs
                nextdeviceQueue.put(taskID)
                
            else: # ending stage
                # outputs = CausalLMOutputWithPast(
                #     loss=tuple_outputs[0],
                #     logits=tuple_outputs[1],
                #     past_key_values=tuple_outputs[2],
                #     hidden_states=tuple_outputs[3],
                #     attentions=tuple_outputs[4],
                # )
                # loss = outputs.loss
                loss = tuple_outputs[0]
                print("[NLL loss={}] stage {} finished task {}".format(loss, device, taskID))
                self.metrics["loss"].append(loss.item())
                
                # if self.setting == 'active':
                if task.require_training:
                    # Backprop on the last stage
                    try:
                        loss.backward()
                        record_time(init_device, 'end', 'backward', timing_info)
                    except Exception as e:
                        logging.error(f"[node {nodeID} | stage {stageID}] Backward error occurred: {e}")
                        # pass
                    
                    # Optimize
                    # self.optimize(nodeID)
                    self.distributed_optimizers[nodeID].step()
                    self.distributed_schedulers[nodeID].step()
                    self.distributed_optimizers[nodeID].zero_grad() # clear gradients
                    print("Stage {} finish backward propagation for task {} !".format(device, taskID))
                    
                # else:
                #     task.hiddens.append(loss)
                #     deviceQueue.put(taskID) # put it back to the queue


    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='model name or path')
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
    
    distributed_llm = DistributedLlama(args)
    distributed_llm.run()
