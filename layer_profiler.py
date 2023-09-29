import time
import psutil
import GPUtil
import argparse
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional
from functools import partial
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BertModel,
    GPT2Model,
    T5Model,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertLayer,
    BertEmbeddings,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
)
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5Block,
)



class LayerProfiler:
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: argparse.Namespace,
        layer_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.config = config
        self.layer_input_length = defaultdict(list)
        self.layer_forward_reach = defaultdict(float)
        self.layer_time_forward = defaultdict(list)
        self.layer_memory_forward = defaultdict(list)
        self.layer_memory_backward = defaultdict(list)
        # self.layer_forward_cpu_utils = defaultdict(list)
        # self.layer_forward_gpu_utils = defaultdict(list)
        # self.layer_backward_cpu_utils = defaultdict(list)
        # self.layer_backward_gpu_utils = defaultdict(list)
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        if layer_name is None:
            self.register_hook(self.model)
        else:
            submodel = getattr(self.model, layer_name)
            self.register_hook(submodel)
            
            
    def get_module_name(self, module: nn.Module):
        return next(name for name, mod in self.model.named_modules() if mod is module)

        
    def take_time_pre_forward(
        self,
        name: str,
        layer: nn.Module,
        module: nn.Module,
        inputs: Tuple[torch.Tensor],
    ):
        self.layer_forward_reach[name] = time.time()
        
        
    def take_time_forward(
        self,
        name: str,
        layer: nn.Module,
        module: nn.Module,
        inputs: Tuple[torch.Tensor],
        outputs: Tuple[torch.Tensor],
        **kwargs,
    ):
        # gpus = GPUtil.getGPUs()
        # gpu_utilization = gpus[0].load if gpus else None
        # self.layer_forward_cpu_utils[name].append(psutil.cpu_percent())
        # self.layer_forward_gpu_utils[name].append(gpu_utilization)
        
        if isinstance(layer, nn.Embedding) and layer.num_embeddings == self.config.vocab_size:
            self.layer_input_length[name].append(inputs[0].shape[1])
            
        self.layer_time_forward[name].append(time.time() - self.layer_forward_reach[name])
        self.layer_memory_forward[name].append(torch.cuda.memory_allocated())

        
    def take_time_backward(
        self, 
        module: nn.Module, 
        grad_input: Tuple[torch.Tensor], 
        grad_output: Tuple[torch.Tensor],
        *args,  # Accept additional arguments
    ):
        name = self.get_module_name(module)
        # gpus = GPUtil.getGPUs()
        # gpu_utilization = gpus[0].load if gpus else None
        # self.layer_backward_cpu_utils[name].append(psutil.cpu_percent())
        # self.layer_backward_gpu_utils[name].append(gpu_utilization)
        self.layer_memory_backward[name].append(torch.cuda.memory_allocated())
            
        
    def register_hook(
        self, 
        module: nn.Module,
        layer_idx: Optional[str] = '',
    ): 
        for name, layer in module.named_children():
            if name.isnumeric() or isinstance(layer, nn.Embedding):
                # print("name: {}, layer: {}".format(name, layer))
                layer_idx = name
            
            if isinstance(layer, (nn.ModuleList, nn.Sequential)):
                # DFS recursion for all modules in the ModuleList
                self.register_hook(layer, layer_idx)
                
            elif isinstance(layer, (BertModel, BertEncoder, BertLayer, BertEmbeddings)):
                # DFS recursion for all modules in the Bert Modules
                self.register_hook(layer, layer_idx) 
                
            elif isinstance(layer, (GPT2Model, GPT2Block)):
                # DFS recursion for all modules in the GPT2Block
                self.register_hook(layer, layer_idx)
                
            elif isinstance(layer, (T5Model, T5Stack, T5Block)):
                # DFS recursion for all modules in the T5 Modules
                self.register_hook(layer, layer_idx)
                
            else:
                if layer_idx:
                    layer_name = layer.__class__.__name__ + '-' + layer_idx
                else:
                    layer_name = layer.__class__.__name__
                layer.register_forward_pre_hook(
                    partial(self.take_time_pre_forward, layer_name, layer)
                )
                layer.register_forward_hook(
                    partial(self.take_time_forward, layer_name, layer)
                )
                if hasattr(layer, 'register_full_backward_hook'):
                    layer.register_full_backward_hook(
                        partial(self.take_time_backward, layer)
                    )
                else:
                    layer.register_backward_hook(
                        partial(self.take_time_backward, layer)
                    )
                self.logger.info(f"Register hook for {layer_name}")



if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--profiling_iterations', type=int, default=1000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output_dir', type=str, default='logging')
    args = parser.parse_args()
    
    model_base_name = args.model_name_or_path.split('/')[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=f"{args.output_dir}/layer_profiler_{model_base_name}.log",
        filemode="w",
    )
    logger = logging.getLogger(__name__)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, config=config).to(device)
    if 'gpt' in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    
    profiler = LayerProfiler(model, config, logger=logger)
    
    input_texts =[
        "Hello, my dog is cute",
        "What a nice day!",
        "I am so happy to see you!",
        "Would you mind helping me with my homework?",
    ]
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)
    
    # Record memory and latency
    pre_memory = 0
    begin = time.time()
    
    for i in range(args.profiling_iterations):
        pre_memory += torch.cuda.memory_allocated()
        try:
            outputs = model(**inputs)
        except:
            inputs['decoder_input_ids'] = inputs['input_ids']
            inputs['decoder_attention_mask'] = inputs['attention_mask']
            outputs = model(**inputs)
        
    end = time.time()
    
    # Determine the maximum key length for proper alignment
    max_key_length = max(len(str(key)) for key in profiler.layer_time_forward)

    # Sort: print the results grouped by layer type
    # No sort: print the results grouped by the order of execution
    sort = True
    if sort:
        # Sort the dictionary based on keys
        sorted_keys = sorted(profiler.layer_time_forward.keys())
    else:
        sorted_keys = profiler.layer_time_forward.keys()

    logger.info(f"pre_memory: {pre_memory / ((1024 ** 2) * args.profiling_iterations)} MB")
    for key in sorted_keys:
        time_val = sum(profiler.layer_time_forward[key]) / args.profiling_iterations
        memory_val = sum(profiler.layer_memory_forward.get(key, 0)) / ((1024 ** 2) * args.profiling_iterations)  # Convert from bytes to MB
        # cpu_val = sum(profiler.layer_forward_cpu_utils[key]) / args.profiling_iterations
        # gpu_val = sum(profiler.layer_forward_gpu_utils[key]) / args.profiling_iterations
        logger.info(f"{key:<{max_key_length}} \t Time: {time_val*1000:.6f} ms \t Memory: {memory_val:.2f} MB")

    # TODO: postprocessing for the results: delta memory, memory ratio, latency ratio, etc.
    logger.info(f"Total time: {(end - begin):.6f} s")
    