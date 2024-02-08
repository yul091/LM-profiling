import sys 
sys.dont_write_bytecode = True
import copy
import warnings
import inspect
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from .llama import LlamaEndingStage
from .dialogpt import GPTEndingStage
from .utils import stages_forward
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamScorer,
    BeamSearchScorer,
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.cache_utils import DynamicCache
from transformers.generation import (
    BeamSearchDecoderOnlyOutput, 
    BeamSearchEncoderDecoderOutput,
    validate_stopping_criteria,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
BeamSearchOutput = Union[
    BeamSearchEncoderDecoderOutput, 
    BeamSearchDecoderOnlyOutput,
]


# @profile
@torch.no_grad()
def greedy_search(
    stages: List[Union[LlamaEndingStage, GPTEndingStage]], 
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    synced_gpus: bool = False,
    **model_kwargs,
):
    # instantiate logits processors
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else stages[-1].generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else stages[-1].generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id, device=stages[-1].device) if eos_token_id is not None else None
    # init attention / hidden states / scores tuples
    scores = None
    
    # keep track of which sequences are already finished
    input_ids = input_ids.to(stages[-1].device)
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=stages[-1].device)
    this_peer_finished = False  # used by synced_gpus only
    step = 0
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
        
        # prepare model inputs
        model_inputs = stages[-1].prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        # forward pass to get next token
        tuple_outputs = stages_forward(stages, model_inputs)
        outputs = CausalLMOutputWithPast(
            loss=tuple_outputs[0],
            logits=tuple_outputs[1],
            past_key_values=tuple_outputs[2],
            hidden_states=tuple_outputs[3],
            attentions=tuple_outputs[4],
        )
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        next_token_logits = outputs.logits[:, -1, :]
        
        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        
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
        step += 1
        print(f"[step {step}] next_tokens: {next_tokens} ({next_tokens.device}), eos_token_id_tensor: {eos_token_id_tensor} ({eos_token_id_tensor.device})")
        print(f"\tunfinished_sequences: {unfinished_sequences} ({unfinished_sequences.device})")
        
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            # print(f"\tunfinished_sequences (before max): {unfinished_sequences} ({unfinished_sequences.device})")
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
        
    return input_ids


@torch.no_grad()
def beam_search(
    stages: List[Union[LlamaEndingStage, GPTEndingStage]],
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
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
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
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
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )

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
        decoder_prompt_len=decoder_prompt_len,
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
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]
    
@torch.no_grad()
def generate(
    stages: List[Union[LlamaEndingStage, GPTEndingStage]],
    inputs: Dict[str, Union[torch.Tensor, Any]],
):
    if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
        synced_gpus = True
    else:
        synced_gpus = False
        
    input_ids = inputs.pop('input_ids')
    
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    stages[-1]._validate_model_class()
    generation_config = copy.deepcopy(stages[-1].generation_config)
    generation_config.num_beams = 3
    generation_config.max_length = 1024
    generation_config.max_new_tokens = 1024
    model_kwargs = generation_config.update(**inputs)
    generation_config.validate()
    stages[-1]._validate_model_kwargs(model_kwargs.copy())
    
    # 2. Set generation parameters if not already defined
    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList()
    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        generation_config.pad_token_id = eos_token_id
    
    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = stages[-1]._prepare_model_inputs(
        input_ids, generation_config.bos_token_id, model_kwargs
    )
    
    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not stages[-1].config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache
        
    accepts_attention_mask = "attention_mask" in set(inspect.signature(stages[-1].forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = stages[-1]._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )
    
    # 5. Prepare 'input_ids' which will be used for auto-regressive generation
    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # 6. Prepare `max_length` depending on other stopping criteria
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    stages[-1]._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
    
    # 7. prepare distribution pre_processing samplers
    logits_processor = stages[-1]._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
    )
    
    # 8. prepare stopping criteria
    stopping_criteria = stages[-1]._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    
    # 9. run greedy search
    return greedy_search(
        stages,
        input_ids,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        synced_gpus=synced_gpus,
        **model_kwargs,
    )
    
    # # 9. prepare beam search scorer
    # beam_scorer = BeamSearchScorer(
    #     batch_size=input_ids.shape[0],
    #     num_beams=generation_config.num_beams,
    #     device=stages[-1].device,
    #     length_penalty=generation_config.length_penalty,
    #     do_early_stopping=generation_config.early_stopping,
    #     num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #     max_length=generation_config.max_length,
    # )
    
    # # 10. interleave input_ids with `num_beams` additional sequences per batch
    # input_ids, model_kwargs = stages[-1]._expand_inputs_for_generation(
    #     input_ids=input_ids,
    #     expand_size=generation_config.num_beams,
    #     is_encoder_decoder=stages[-1].config.is_encoder_decoder,
    #     **model_kwargs,
    # )
    
    # # 11. run beam search
    # return beam_search(
    #     stages, 
    #     input_ids,
    #     beam_scorer,
    #     logits_processor=logits_processor,
    #     stopping_criteria=stopping_criteria,
    #     pad_token_id=generation_config.pad_token_id,
    #     eos_token_id=generation_config.eos_token_id,
    #     output_scores=generation_config.output_scores,
    #     return_dict_in_generate=generation_config.return_dict_in_generate,
    #     synced_gpus=synced_gpus,
    #     **model_kwargs,
    # )
    
    
    
if __name__ == '__main__':
    import time
    
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
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    config = AutoConfig.from_pretrained(model_name_or_path, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs['labels'] = inputs['input_ids'].clone()
    encoder_input_ids = inputs["input_ids"].clone()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        config=config, 
        token=access_token,
        device_map='auto',
        # torch_dtype="auto"
    )

    # Distributed generation (standard pipeline)
    start = time.time()
    generate_ids = model.generate(**inputs, do_sample=False)
    end = time.time()
    if model.config.is_encoder_decoder:
        output_ids = generate_ids
    else:
        output_ids = []
        for i in range(generate_ids.shape[0]):
            output_ids.append(generate_ids[i][len(encoder_input_ids[i]):])
        output_ids = torch.stack(output_ids, dim=0)
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f'generation overhead {end - start}: \n{responses}')
    
    
    # num_stages = 8
    # timing_info = defaultdict(list)
    # stages = get_stages(
    #     config, 
    #     access_token,
    #     model_name_or_path,
    #     num_stages, 
    #     timing_info=timing_info,
    # )
    
    # start = time.time()
    # generate_ids = generate(stages, inputs)
    # end = time.time()
    # if model.config.is_encoder_decoder:
    #     output_ids = generate_ids
    # else:
    #     output_ids = []
    #     for i in range(generate_ids.shape[0]):
    #         output_ids.append(generate_ids[i][len(encoder_input_ids[i]):])
    #     output_ids = torch.stack(output_ids, dim=0)
    # responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print(f'generation overhead {end - start}: \n{responses}')