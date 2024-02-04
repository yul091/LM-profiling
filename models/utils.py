import sys 
sys.dont_write_bytecode = True
import copy
import warnings
from collections import defaultdict
from collections.abc import Mapping
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from llama import (
    LlamaStartingStage,
    LlamaIntermediateStage,
    LlamaEndingStage,
)
from transformers import (
    LlamaConfig, 
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamScorer,
    BeamSearchScorer,
)
from transformers.cache_utils import DynamicCache
from transformers.generation import (
    BeamSearchDecoderOnlyOutput, 
    BeamSearchEncoderDecoderOutput,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
BeamSearchOutput = Union[
    BeamSearchEncoderDecoderOutput, 
    BeamSearchDecoderOnlyOutput,
]


def get_stages(
    config: LlamaConfig,
    token: str,
    model_name_or_path: str,
    num_stages: int,
    hidden_layers_assignments: List[int] = None,
    init_device: int = 0,
    timing_info: dict = None,
) -> List[Union[LlamaStartingStage, LlamaIntermediateStage, LlamaEndingStage]]:
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
    
    for stage_id in range(num_stages):
        # Overwrite num_hidden_layers with the number for this stage
        config = copy.deepcopy(config)
        config.pad_token_id = config.eos_token_id
        config.num_hidden_layers = hidden_layers_assignments[stage_id]
        device = init_device + stage_id
        layer_ids = list(range(stage_id * config.num_hidden_layers, (stage_id + 1) * config.num_hidden_layers))
        if stage_id == 0:
            # Starting stage
            stage = LlamaStartingStage(
                config=config,
                device=device, 
                layer_ids=layer_ids,
                pretrained_model=pretrained_model,
                timing_info=timing_info,
            )
        elif stage_id == num_stages - 1:
            # Ending stage
            stage = LlamaEndingStage(
                config=config,
                device=device, 
                layer_ids=layer_ids,
                pretrained_model=pretrained_model,
                timing_info=timing_info,
            )
            # Set pad_token_id to eos_token_id because GPT/Llama does not have a PAD token
            stage.generation_config.pad_token_id = stage.generation_config.eos_token_id
        else:
            # Intermediate stage
            stage = LlamaIntermediateStage(
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
    stages: List[LlamaStartingStage], 
    inputs: Dict[str, Union[torch.Tensor, Any]],
):
    labels = inputs.get('labels', None)
    labels = labels.to(stages[-1].device) if labels is not None else None
    for i, stage in enumerate(stages):
        # Prepare inputs
        if i == 0:
            batch_inputs = _prepare_inputs(inputs, device=stage.device)
            outputs = stage(**batch_inputs)
        else:
            batch_inputs = _prepare_inputs(outputs, device=stage.device) 
            outputs = stage(
                hidden_states=batch_inputs[0],
                past_key_values=batch_inputs[1],
                all_hidden_states=batch_inputs[2],
                all_self_attns=batch_inputs[3],
                position_ids=batch_inputs[4],
                attention_mask=batch_inputs[5],
                labels=labels,
            )
            
        # Only change the input_ids/hidden_states, keep the rest the same as in the original inputs
        # print(f"Forward pass for stage {i} on device {stage.device}")
        # for k, v in batch_inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.device}")
        #     elif isinstance(v, tuple):
        #         for i, t in enumerate(v):
        #             print(f"{k}[{i}]: {t.device}")
        #     else:
        #         print(f"{k}: {type(v)}")   
        
    return outputs


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

@ torch.no_grad()
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
        model_kwargs = stages[-1]._update_model_kwargs_for_generation(
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

@ torch.no_grad()
def beam_search(
    stages: List[LlamaEndingStage],
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    # instantiate logits processors
    min_length = min_length if min_length is not None else 0
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList([
        MinLengthLogitsProcessor(min_length, eos_token_id=stages[-1].generation_config.eos_token_id),
    ])
    max_length = max_length if max_length is not None else 128
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList([
        MaxLengthCriteria(max_length=max_length)
    ])
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else stages[-1].generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else stages[-1].generation_config.eos_token_id 
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else stages[-1].generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else stages[-1].generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else stages[-1].generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else stages[-1].generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and stages[-1].config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = stages[-1].prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Forward pass
        tuple_outputs = stages_forward(stages, model_inputs)
        outputs = CausalLMOutputWithPast(
            loss=tuple_outputs[0],
            logits=tuple_outputs[1],
            past_key_values=tuple_outputs[2],
            hidden_states=tuple_outputs[3],
            attentions=tuple_outputs[4],
        )
        
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        input_ids = input_ids.to(next_token_logits.device) # put on the output device
        beam_scores = beam_scores.to(next_token_logits.device)  # put on the output device
        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores_processed)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if stages[-1].config.is_encoder_decoder else (outputs.attentions,)
                )
                if stages[-1].config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if stages[-1].config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = stages[-1]._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=stages[-1].config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = stages[-1]._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if stages[-1].config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]



if __name__ == '__main__':
    import time
    import pdb
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from fastchat.model import get_conversation_template
    
    # def process(msg: str, model_path: str):
    #     conv = get_conversation_template(model_path)
    #     conv.append_message(conv.roles[0], msg)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    #     return prompt
    
    texts = [
        "Hello, my dog is cute",
        "What is your name?",
        "I recently bought a new car. It is a Tesla, and it is very fast."
    ]
    print("Queries: ", texts)
    
    access_token = "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"
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    config = AutoConfig.from_pretrained(model_name_or_path, token=access_token)
    # config.use_cache = False # try not using cache
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # prompts = [process(text, model_name_or_path) for text in texts]
    # print("Prompts: ", prompts)
    
    # Test Causal Language Modeling
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs['labels'] = inputs['input_ids'].clone()
    encoder_input_ids = inputs["input_ids"].clone()
    # print(encoder_inputs)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        config=config, 
        token=access_token,
        device_map='auto',
        # torch_dtype="auto"
    )
    print(model.model.embed_tokens.weight)
    # generate_ids = model.generate(**inputs, do_sample=False)
    # if model.config.is_encoder_decoder:
    #     output_ids = generate_ids
    # else:
    #     output_ids = []
    #     for i in range(generate_ids.shape[0]):
    #         output_ids.append(generate_ids[i][len(encoder_input_ids[i]):])
    #     output_ids = torch.stack(output_ids, dim=0)
    # responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print(responses)
    outputs = model(**inputs)
    loss = outputs.loss
    logits = outputs.logits
    print(f'loss: {loss}, logits: {logits}')
    
    num_stages = 4
    timing_info = defaultdict(list)
    stages = get_stages(
        config, 
        access_token,
        model_name_or_path,
        num_stages, 
        timing_info=timing_info,
    )
    print(stages[0].embed_tokens.weight)
    outputs = stages_forward(stages, inputs)
    loss = outputs[0]
    logits = outputs[1]
    print(f'loss: {loss}, logits: {logits}')
    # loss.backward()
    # timing_info['0_end'].append((time.time(), 'backward'))
    # print(timing_info)
    
    # # Test Greedy Search
    # # input_ids = inputs.pop('input_ids')
    # # outputs = greedy_search(
    # #     stages, 
    # #     input_ids,
    # #     logits_processor=None,
    # #     stopping_criteria=None,
    # #     pad_token_id=None,
    # #     eos_token_id=None,
    # #     synced_gpus=False,
    # #     **inputs,
    # # )
    # # responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # # print(responses)
    
    # # Instantiate beam scorer
    # input_ids = inputs.pop('input_ids')
    # generation_config = copy.deepcopy(stages[-1].generation_config)
    # generation_config.num_beams = 3
    # model_kwargs = generation_config.update(**inputs)
    # beam_scorer = BeamSearchScorer(
    #     batch_size=input_ids.shape[0],
    #     num_beams=generation_config.num_beams,
    #     device=stages[-1].device,
    #     length_penalty=generation_config.length_penalty,
    #     do_early_stopping=generation_config.early_stopping,
    #     num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #     max_length=generation_config.max_length,
    # )
    # # Interleave input_ids with `num_beams` additional sequences per batch
    # input_ids, model_kwargs = stages[-1]._expand_inputs_for_generation(
    #     input_ids=input_ids,
    #     expand_size=generation_config.num_beams,
    #     is_encoder_decoder=stages[-1].config.is_encoder_decoder,
    #     **model_kwargs,
    # )
    # # Run beam search
    # outputs = beam_search(
    #     stages, 
    #     input_ids,
    #     beam_scorer,
    #     pad_token_id=generation_config.pad_token_id,
    #     eos_token_id=generation_config.eos_token_id,
    #     output_scores=generation_config.output_scores,
    #     return_dict_in_generate=generation_config.return_dict_in_generate,
    #     **model_kwargs,
    # )
    # # if stages[-1].config.is_encoder_decoder:
    # #     output_ids = outputs
    # # else:
    # #     output_ids = []
    # #     for i in range(input_ids.shape[0]):
    # #         output_ids.append(outputs[i][len(encoder_input_ids[i]):])
    # #     output_ids = torch.cat(output_ids, dim=0)
        
    # responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(responses)