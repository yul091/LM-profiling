import time
import argparse
import logging
from typing import List, Tuple, Dict, Union, Optional
from functools import partial
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertLayer,
    BertIntermediate,
    BertOutput,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
)
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5Block,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
)



class LayerProfiler:
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: argparse.Namespace,
        layer_name: Optional[str] = None,
    ):
        self.model = model
        self.layer_name =  layer_name
        self.config = config
        self.layer_time_reach = {}
        self.layer_time_cost = {}
        self.layer_memory = {}
        self.register_hook(self.model, depth=0)
        
        
    def take_time_pre(
        self,
        name: str,
        layer: nn.Module,
        module: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ):
        self.layer_time_reach[name] = time.time()
        
        
    def take_time(
        self,
        name: str,
        layer: nn.Module,
        module: nn.Module,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ):
        if self.config.debug:
            logger.info("[DEBUG]torch.cuda.memory_allocated():", torch.cuda.memory_allocated())
        if name in self.layer_time_cost:
            self.layer_time_cost[name] += time.time() - self.layer_time_reach[name]
            self.layer_memory[name] += torch.cuda.memory_allocated()
        else:
            self.layer_time_cost[name] = time.time() - self.layer_time_reach[name]
            self.layer_memory[name] = torch.cuda.memory_allocated()
        
        
    def register_hook(
        self, 
        module: nn.Module,
        depth: Optional[int] = None,
    ):
        if depth is None:
            depth = 0
            
        for name, layer in module.named_children():
            if isinstance(layer, nn.ModuleList):
                # DFS recursion for all modules in the ModuleList
                self.register_hook(layer, depth + 1)
            elif isinstance(layer, nn.Sequential):
                # DFS recursion for all modules in the Sequential
                self.register_hook(layer, depth + 1)
            elif isinstance(layer, (BertEmbeddings, BertEncoder, BertPooler, BertLayer, BertIntermediate, BertOutput)):
                # DFS recursion for all modules in the Bert Modules
                self.register_hook(layer, depth + 1) 
            elif isinstance(layer, GPT2Block):
                # DFS recursion for all modules in the GPT2Block
                self.register_hook(layer, depth + 1)
            elif isinstance(layer, (T5Stack, T5Block, T5LayerSelfAttention, T5LayerCrossAttention)):
                # DFS recursion for all modules in the T5 Modules
                self.register_hook(layer, depth + 1)
            else:
                layer.register_forward_pre_hook(
                    partial(self.take_time_pre, str(type(layer)) + ':' + str(depth) + ':' + name, layer)
                )
                layer.register_forward_hook(
                    partial(self.take_time, str(type(layer)) + ':' + str(depth) + ':' + name, layer)
                )
                logger.info(f"Register hook for {name} ({type(layer)}) at depth {depth}")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--profiling_iterations', type=int, default=1000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sort', action='store_true')
    args = parser.parse_args()
    
    model_base_name = args.model_name_or_path.split('/')[-1]
    logging.basicConfig(
        format="%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=f"layer_profiler_{model_base_name}.log",
        filemode="w",
    )
    logger = logging.getLogger(__name__)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.debug = args.debug
    model = AutoModel.from_pretrained(args.model_name_or_path, config=config).to(device)
    if 'gpt' in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    
    profiler = LayerProfiler(model, config)
    
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
    max_key_length = max(len(str(key)) for key in profiler.layer_time_cost)

    # Sort: print the results grouped by layer type
    # No sort: print the results grouped by the order of execution
    sort = args.sort
    if sort:
        # Sort the dictionary based on keys
        sorted_keys = sorted(profiler.layer_time_cost.keys())
    else:
        sorted_keys = profiler.layer_time_cost.keys()

    logger.info(f"pre_memory: {pre_memory / ((1024 ** 2) * args.profiling_iterations)} MB")
    for key in sorted_keys:
        time_val = profiler.layer_time_cost[key] / args.profiling_iterations
        memory_val = profiler.layer_memory.get(key, 0) / ((1024 ** 2) * args.profiling_iterations)  # Convert from bytes to MB
        logger.info(f"{key:<{max_key_length}} \t Time: {time_val*1000:.6f} ms \t Memory: {memory_val:.2f} MB")

    # TODO: postprocessing for the results: delta memory, memory ratio, latency ratio, etc.
    logger.info(f"Total time: {(end - begin):.6f} s")
    