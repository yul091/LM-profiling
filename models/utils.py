import sys 
sys.dont_write_bytecode = True
import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch.nn as nn
from llama import (
    LlamaStartingStage,
    LlamaIntermediateStage,
    LlamaEndingStage,
)
from dialogpt import (
    GPTStartingStage,
    GPTIntermediateStage,
    GPTEndingStage,
)
from transformers import (
    LlamaConfig, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


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
        elif isinstance(data, DynamicCache):
            data.key_cache = _prepare_input(data.key_cache, device)
            data.value_cache = _prepare_input(data.value_cache, device)
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


def get_stages(
    config: LlamaConfig,
    token: str,
    model_name_or_path: str,
    num_stages: int,
    hidden_layers_assignments: List[int] = None,
    init_device: int = 0,
    timing_info: dict = None,
) -> List[Union[GPTEndingStage, LlamaEndingStage]]:
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
    
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        config=config, 
        token=token,
    )
    
    pipeline_stages = []
    totoal_params = 0
    start_stage_class = GPTStartingStage if 'gpt' in model_name_or_path.lower() else LlamaStartingStage
    intermediate_stage_class = GPTIntermediateStage if 'gpt' in model_name_or_path.lower() else LlamaIntermediateStage
    end_stage_class = GPTEndingStage if 'gpt' in model_name_or_path.lower() else LlamaEndingStage
    
    for stage_id in range(num_stages):
        # Overwrite num_hidden_layers with the number for this stage
        config = copy.deepcopy(config)
        config.pad_token_id = config.eos_token_id
        config.num_hidden_layers = hidden_layers_assignments[stage_id]
        device = init_device + stage_id
        layer_ids = list(range(stage_id * config.num_hidden_layers, (stage_id + 1) * config.num_hidden_layers))
        if stage_id == 0:
            # Starting stage
            stage = start_stage_class(
                config=config,
                device=device, 
                layer_ids=layer_ids,
                pretrained_model=pretrained_model,
                timing_info=timing_info,
            )
        elif stage_id == num_stages - 1:
            # Ending stage
            stage = end_stage_class(
                config=config,
                device=device, 
                layer_ids=layer_ids,
                pretrained_model=pretrained_model,
                timing_info=timing_info,
            )
            # Set pad_token_id to eos_token_id because Llama / GPT does not have a PAD token
            stage.generation_config.pad_token_id = stage.generation_config.eos_token_id
        else:
            # Intermediate stage
            stage = intermediate_stage_class(
                config=config,
                device=device, 
                layer_ids=layer_ids,
                pretrained_model=pretrained_model,
                timing_info=timing_info,
            )
            
        print(f"Put stage {stage.__class__.__name__} ({stage.num_parameters()} parameters) on device {device}")
        totoal_params += stage.num_parameters()
        pipeline_stages.append(stage)
            
    return pipeline_stages


def stages_forward(
    stages: List[Union[GPTEndingStage, LlamaEndingStage]], 
    inputs: Dict[str, Union[torch.Tensor, Any]],
):
    labels = inputs.get('labels', None)
    labels = labels.to(stages[-1]._device) if labels is not None else None
    for i, stage in enumerate(stages):
        print(f"Forward pass for stage {i} on device {stage._device}")
        if i == 0:
            batch_inputs = _prepare_inputs(inputs, device=stage._device)
            outputs = stage(**batch_inputs)
        else:
            batch_inputs = _prepare_inputs(outputs, device=stage._device) 
            if "llama" in stage.__class__.__name__.lower():
                outputs = stage(
                    hidden_states=batch_inputs[0],
                    past_key_values=batch_inputs[1],
                    all_hidden_states=batch_inputs[2],
                    all_self_attns=batch_inputs[3],
                    position_ids=batch_inputs[4],
                    attention_mask=batch_inputs[5],
                    labels=labels,
                )
            elif "gpt" in stage.__class__.__name__.lower():
                outputs = stage(
                    hidden_states=batch_inputs[0],
                    attention_mask=batch_inputs[1],
                    head_mask=batch_inputs[2],
                    encoder_hidden_states=batch_inputs[3],
                    encoder_attention_mask=batch_inputs[4],
                    all_hidden_states=batch_inputs[5],
                    all_self_attentions=batch_inputs[6],
                    all_cross_attentions=batch_inputs[7],
                    output_shape=batch_inputs[8],
                    labels=labels,
                )
            else:
                raise ValueError(f"Unknown model type: {stage.__class__.__name__}")
        
    return outputs


def stages_decoding(
    stages: List[Union[GPTEndingStage, LlamaEndingStage]], 
    inputs: Dict[str, Union[torch.Tensor, Any]],
    labels: Dict[str, Union[torch.Tensor, Any]],
):
    # Step 1: prepare inputs
    # For clm, we need to mask the sentence (-100) and add to the labels
    new_inputs = inputs.copy()
    for k in ['input_ids', 'attention_mask']:
        new_inputs[k] = torch.cat((inputs[k], labels[k]), dim=1)
    new_labels = torch.cat(
        (-100 * torch.ones_like(inputs["input_ids"]), labels["input_ids"]), 
        dim=1,
    )
    new_inputs['labels'] = new_labels
    return stages_forward(stages, new_inputs)
    
    
def compute_nll_loss(
    model: LlamaEndingStage,
    inputs: Dict[str, Union[torch.Tensor, Any]], 
    labels: Dict[str, Union[torch.Tensor, Any]],
    concate_keys: List[str] = ['input_ids', 'attention_mask'],
) -> torch.Tensor:
    # For clm, we need to mask the sentence (-100) and add to the labels
    new_inputs = inputs.copy()
    for k in concate_keys:
        new_inputs[k] = torch.cat((inputs[k], labels[k]), dim=1)
    new_labels = torch.cat(
        (-100 * torch.ones_like(inputs["input_ids"]), labels["input_ids"]), 
        dim=1,
    )
    outputs = model(**new_inputs, labels=new_labels)
    return outputs.loss


if __name__ == '__main__':
    import time
    import pdb
    
    texts = [
        "Hello, my dog is cute",
        "What is your name?",
        "I recently bought a new car. It is a Tesla, and it is very fast.",
    ]
    references = [
        "\nI'm glad you think so! Dogs are always a joy to have around, and they can bring so much happiness and companionship to our lives. Is your dog a specific breed, or a mix of breeds? Do you have any funny or interesting stories about your dog that you'd like to share?", 
        '\n\nMy name is Sherlock Holmes.\n\nWhat is your occupation?\n\nI am a consulting detective.\n\nWhat is your address?\n\nI do not have an address. I am a wanderer of the streets of London.\n\nWhat is your favorite food?\n\nI do not have a favorite food. I am a man of simple tastes and eat whatever is available.\n\nWhat is your favorite drink?\n\nI do not drink alcohol. I find it to be a hindrance to my work.\n\nWhat is your favorite hobby?\n\nI do not have a favorite hobby. I am too busy solving crimes to have time for hobbies.\n\nWhat is your favorite book?\n\nI do not have a favorite book. I am more interested in solving crimes than reading books.\n\nWhat is your favorite music?\n\nI do not have a favorite type of music. I am too busy solving crimes to listen to music.\n\nWhat is your favorite sport?\n\nI do not have a favorite sport. I am too busy solving crimes to have time for sports.\n\nWhat is your favorite animal?\n\nI do not have a favorite animal. I am too busy solving crimes to have time for pets.\n\nWhat is your favorite color?\n\nI do not have a favorite color. I am too busy solving crimes to have time to think about such trivial matters.\n\nWhat is your favorite place to visit?\n\nI do not have a favorite place to visit. I am too busy solving crimes to have time to travel.\n\nWhat is your favorite thing to do?\n\nI do not have a favorite thing to do. I am too busy solving crimes to have time for leisure activities.\n\nWhat is your favorite quote?\n\n"Elementary, my dear Watson."', 
        'I can go from 0 to 60 in just 3.2 seconds. It is also very environmentally friendly, as it runs on electricity and produces zero emissions. I am very happy with my new car and I think it will be a great addition to my family.\n\nI am writing to tell you about my new car because I am very excited about it. I have always been interested in technology and innovation, and a Tesla is the perfect combination of both. The car is equipped with advanced technology, including a large touchscreen display and a sophisticated autopilot system. It also has a range of safety features, such as automatic emergency braking and lane departure warning.\n\nI am also impressed with the design of the car. It has a sleek and modern look, and it is very easy to drive. The car is also very spacious, with plenty of room for passengers and cargo. I am confident that it will be a reliable and comfortable vehicle for many years to come.\n\nOverall, I am very pleased with my new Tesla and I think it will be a great addition to my family. I am excited to take it on long road trips and to show it off to my friends and family. I am sure that it will provide many years of safe and enjoyable driving.',
    ]
    print("Queries: ", texts)
    
    access_token = "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"
    # model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    model_name_or_path = "microsoft/DialoGPT-small"
    config = AutoConfig.from_pretrained(model_name_or_path, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        config=config, 
        token=access_token,
        # device_map='auto',
        # torch_dtype="auto"
    )
    
    num_stages = 4
    timing_info = defaultdict(list)
    stages = get_stages(
        config, 
        access_token,
        model_name_or_path,
        num_stages, 
        timing_info=timing_info,
    )
    
    # Test Causal Language Modeling (CLM)
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs['labels'] = inputs['input_ids'].clone()
    encoder_input_ids = inputs["input_ids"].clone()
    
    start = time.time()
    outputs = model(**inputs)
    loss = outputs.loss
    logits = outputs.logits
    end = time.time()
    print(f'loss: {loss}, logits: {logits.shape}')
    print(f'forward overhead {end - start}')

    start = time.time()
    outputs = stages_forward(stages, inputs)
    loss = outputs[0]
    logits = outputs[1]
    end = time.time()
    print(f'loss: {loss}, logits: {logits.shape}')
    print(f'forward overhead {end - start}')
    # loss.backward()
    # timing_info['0_end'].append((time.time(), 'backward'))
    # print(timing_info)
    
    # Test decoding nll loss
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    labels = tokenizer(references, return_tensors="pt", padding=True)
    start = time.time()
    loss = compute_nll_loss(model, inputs, labels)
    end = time.time()
    print(f'nll loss: {loss}, overhead: {end - start}')
    
    start = time.time()
    outputs = stages_decoding(stages, inputs, labels)
    loss = outputs[0]
    end = time.time()
    print(f'nll loss: {loss}, overhead: {end - start}')
    loss.backward()
    timing_info['0_end'].append((time.time(), 'backward'))
    print(timing_info)