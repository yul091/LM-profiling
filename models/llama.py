import copy
import time
from functools import partial
from collections import defaultdict
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, FloatTensor
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.utils import logging 

logger = logging.get_logger(__name__)



class LlamaStartingStage(LlamaModel):
    def __init__(self, config: LlamaConfig, device: int, timing_info: dict = None):
        # Explicitly initialize LlamaModel with its expected arguments
        super().__init__(config)
        self._device = device
        self.register_full_backward_hook(partial(self.backward_hook, timing_info=timing_info))
        # Put the model on the device
        self.to(device)
    
    # Add a method to profile the backward pass
    def backward_hook(
        self, 
        module: nn.Module, 
        grad_input: Tuple[Tensor], 
        grad_output: Tuple[Tensor], 
        timing_info: dict = None,
    ):
        # print(f"Backward pass started for {module}")
        start_time = time.time()
        if f"{self._device+1}_start" in timing_info:
            timing_info[f"{self._device+1}_end"].append((start_time, "backward"))
        timing_info[f"{self._device}_start"].append((start_time, "backward"))
        
    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return outputs.last_hidden_state
        
    
    
class LlamaIntermediateStage(LlamaModel):
    def __init__(self, config: LlamaConfig, device: int, timing_info: dict = None):
        super().__init__(config)
        self._device = device
        self.register_full_backward_hook(partial(self.backward_hook, timing_info=timing_info))
        # Put the model on the device
        self.to(device)
    
    # Add a method to profile the backward pass
    def backward_hook(
        self, 
        module: nn.Module, 
        grad_input: Tuple[Tensor], 
        grad_output: Tuple[Tensor], 
        timing_info: dict = None,
    ):
        # print(f"Backward pass started for {module}")
        start_time = time.time()
        if f"{self._device+1}_start" in timing_info:
            timing_info[f"{self._device+1}_end"].append((start_time, "backward"))
        timing_info[f"{self._device}_start"].append((start_time, "backward"))
        
    def forward(
        self,
        hidden_states: LongTensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        outputs = super().forward(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        return outputs.last_hidden_state
        
    

class LlamaEndingStage(LlamaForCausalLM):
    
    def __init__(self, config: LlamaConfig, device: int, timing_info: dict = None):
        super().__init__(config)
        self._device = device
        self.register_full_backward_hook(partial(self.backward_hook, timing_info=timing_info))
        # Put the model on the device
        self.to(device)
    
    # Add a method to profile the backward pass
    def backward_hook(
        self, 
        module: nn.Module, 
        grad_input: Tuple[Tensor], 
        grad_output: Tuple[Tensor], 
        timing_info: dict = None,
    ):
        # print(f"Backward pass started for {module}")
        start_time = time.time()
        if f"{self._device+1}_start" in timing_info:
            timing_info[f"{self._device+1}_end"].append((start_time, "backward"))
        timing_info[f"{self._device}_start"].append((start_time, "backward"))
        
    def forward(
        self,
        hidden_states: LongTensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        outputs = super().forward(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        return outputs.loss
    
    


def get_stages(
    config: LlamaConfig,
    num_stages: int,
    hidden_layers_assignments: List[int] = None,
    init_device: int = 0,
    timing_info: dict = None,
):
    """
    start_stage (stage_id == 0): nn.Embeddings + [LlamaDecoderLayer] * N_s
    intermediate_stage (0 < stage_id < num_stages - 1): [LlamaDecoderLayer] * N_i
    end_stage (stage_id == num_stages - 1): [LlamaDecoderLayer] * N_e + BertPreTrainingHeads

    N_s, N_i, N_e: the number of hidden layers (LlamaDecoderLayers) for each stage
    """
    assert num_stages > 2, 'At least 3 stages are required.'
    
    if hidden_layers_assignments is None:
        """
        Assign the number of hidden layers (BertLayers) so that
        the following are satisfied: 
            N_e <= N_s <= N_i
        """
        hidden_layers_assignments = [0] * num_stages
        for i in range(config.num_hidden_layers):
            hidden_layers_assignments[-((i + 2) % num_stages)] += 1
    assert len(hidden_layers_assignments) == num_stages
    
    pipeline_stages = []
    totoal_params = 0
    for stage_id in range(num_stages):
        # Overwrite num_hidden_layers with the number for this stage
        config = copy.deepcopy(config)
        config.num_hidden_layers = hidden_layers_assignments[stage_id]
        device = init_device + stage_id
        if stage_id == 0:
            # Starting stage
            stage = LlamaStartingStage(config, device, timing_info=timing_info)
        elif stage_id == num_stages - 1:
            # Ending stage
            stage = LlamaEndingStage(config, device, timing_info=timing_info)
        else:
            # Intermediate stage
            stage = LlamaIntermediateStage(config, device, timing_info=timing_info)
            
        print(f"Put stage {stage.__class__.__name__} ({stage.num_parameters()} parameters) on device {device}")
        totoal_params += stage.num_parameters()
        pipeline_stages.append(stage)
            
    return pipeline_stages


def stages_forward(
    stages: List[LlamaStartingStage], 
    inputs: Dict[str, Union[Tensor, Any]],
):
    # Copy the inputs to avoid modifying the original inputs
    inputs = copy.deepcopy(inputs)
    for i, stage in enumerate(stages):
        # Prepare inputs
        inputs = {k: v.to(stage.device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
        if i == 0:
            hidden = inputs.pop('input_ids').to(stage.device)
        else:
            hidden = hidden.to(stage.device)
        # Only change the input_ids/hidden_states, keep the rest the same as in the original inputs
        hidden = stage(hidden, **inputs)
        
    return hidden
        



if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoConfig
    
    texts = [
        "Hello, my dog is cute",
        "What is your name?",
        "I recently bought a new car. It is a Tesla, and it is very fast."
    ]
    
    access_token = "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"
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    config = AutoConfig.from_pretrained(model_name_or_path, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test Causal Language Modeling
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs['labels'] = inputs['input_ids'].clone()
    print(inputs)
    
    num_stages = 4
    timing_info = defaultdict(list)
    stages = get_stages(config, num_stages, timing_info=timing_info)
    
    loss = stages_forward(stages, inputs)
    print(loss)
    
    loss.backward()
    
    timing_info['0_end'].append((time.time(), 'backward'))
    print(timing_info)