import copy
import time
from functools import partial
from collections import defaultdict
from collections.abc import Mapping
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, FloatTensor
from transformers import (
    LlamaConfig, 
    LlamaModel, 
    LlamaForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
from transformers.utils import logging, ModelOutput 

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
        
        return outputs
    
    


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
        config.pad_token_id = config.eos_token_id
        config.num_hidden_layers = hidden_layers_assignments[stage_id]
        device = init_device + stage_id
        if stage_id == 0:
            # Starting stage
            stage = LlamaStartingStage(config, device, timing_info=timing_info)
        elif stage_id == num_stages - 1:
            # Ending stage
            stage = LlamaEndingStage(config, device, timing_info=timing_info)
            # Set pad_token_id to eos_token_id because GPT/Llama does not have a PAD token
            stage.generation_config.pad_token_id = stage.generation_config.eos_token_id
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
    for i, stage in enumerate(stages):
        # Prepare inputs
        if i == 0:
            batch_inputs = _prepare_inputs(inputs, device=stage.device)
            hidden = batch_inputs.pop('input_ids')
        else:
            batch_inputs = _prepare_inputs(batch_inputs, device=stage.device)
            hidden = hidden.to(stage.device)
        # Only change the input_ids/hidden_states, keep the rest the same as in the original inputs
        # print(f"Forward pass for stage {i} on device {stage.device}")
        # print(f"Hidden device: ", hidden.device)
        # for k, v in batch_inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.device}")
        #     elif isinstance(v, tuple):
        #         for i, t in enumerate(v):
        #             print(f"{k}[{i}]: {t.device}")
        #     else:
        #         print(f"{k}: {type(v)}")
        hidden = stage(hidden, **batch_inputs)
        
    return hidden


def _prepare_input(
    data: Union[torch.Tensor, Any],
    device: torch.device = 'cuda',
) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_prepare_input(v, device) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": device}
            return data.to(**kwargs)
        return data

def _prepare_inputs(
    inputs: Dict[str, Union[torch.Tensor, Any]],
    device: torch.device = 'cuda',
) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    new_inputs = _prepare_input(inputs, device=device)
    if len(new_inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it."
        )
    return new_inputs


def _extract_past_from_model_output(outputs: ModelOutput, standardize_cache_format: bool = False):
    past_key_values = None
    if "past_key_values" in outputs:
        past_key_values = outputs.past_key_values
    elif "mems" in outputs:
        past_key_values = outputs.mems
    elif "past_buckets_states" in outputs:
        past_key_values = outputs.past_buckets_states
    return past_key_values


def update_model_kwargs_for_generation(
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
) -> Dict[str, Any]:
    # update past_key_values
    model_kwargs["past_key_values"] = _extract_past_from_model_output(
        outputs, standardize_cache_format=standardize_cache_format
    )
    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    return model_kwargs


def greedy_search(
    stages: List[LlamaEndingStage], 
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    synced_gpus: bool = False,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    **model_kwargs,
):
    # Instantiate logits processors
    min_length = min_length if min_length is not None else 0
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList([
        MinLengthLogitsProcessor(min_length, eos_token_id=stages[-1].generation_config.eos_token_id),
    ])
    max_length = max_length if max_length is not None else 128
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList([
        MaxLengthCriteria(max_length=max_length)
    ])
    pad_token_id = pad_token_id if pad_token_id is not None else stages[-1].generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else stages[-1].generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id) if eos_token_id is not None else None
    # Init attention / hidden states / scores tuples
    scores = None
    
    # Keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long)
    this_peer_finished = False  # used by synced_gpus only
  
    while True:
        # Prepare inputs
        model_inputs = stages[-1].prepare_inputs_for_generation(input_ids, **model_kwargs)
        # Forward pass
        outputs = stages_forward(stages, model_inputs)
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        # Get logits
        next_token_logits = outputs.logits[:, -1, :]
        input_ids = input_ids.to(next_token_logits.device) 
        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # Put on the output device
        unfinished_sequences = unfinished_sequences.to(next_tokens.device)
        eos_token_id_tensor = eos_token_id_tensor.to(next_tokens.device) if eos_token_id_tensor is not None else None
        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=stages[-1].config.is_encoder_decoder
        )
        
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
    return input_ids


# Example usage:
# Assuming you have defined your stages and config
# starting_stage, intermediate_stages, ending_stage = get_stages(config, num_stages, hidden_layers_assignments, init_device, timing_info)
# input_ids = torch.tensor([[your_initial_token_id]])  # Replace 'your_initial_token_id' with your actual initial token ID
# generated_sequence = generate(starting_stage, intermediate_stages, ending_stage, input_ids)
# print("Generated sequence:", generated_sequence)




if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoConfig
    
    texts = [
        "Hello, my dog is cute",
        "What is your name?",
        "I recently bought a new car. It is a Tesla, and it is very fast."
    ]
    print("Queries: ", texts)
    
    access_token = "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"
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    config = AutoConfig.from_pretrained(model_name_or_path, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test Causal Language Modeling
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs['labels'] = inputs['input_ids'].clone()
    # print(inputs)
    
    num_stages = 4
    timing_info = defaultdict(list)
    stages = get_stages(config, num_stages, timing_info=timing_info)
    
    # outputs = stages_forward(stages, inputs)
    # loss = outputs.loss
    # print(loss)
    # loss.backward()
    # timing_info['0_end'].append((time.time(), 'backward'))
    # print(timing_info)
    
    
    # Test Greedy Search
    input_ids = inputs.pop('input_ids')
    outputs = greedy_search(
        stages, 
        input_ids,
        logits_processor=None,
        stopping_criteria=None,
        pad_token_id=None,
        eos_token_id=None,
        synced_gpus=False,
        **inputs,
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(responses)