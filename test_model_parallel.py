import os
import sys
sys.dont_write_bytecode = True
import time
import math
from tqdm import tqdm
import numpy as np
import logging
from functools import partial
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from typing import Optional, Tuple, List, Union
from torch.cuda.amp import GradScaler
from transformers import (
    BertPreTrainedModel, 
    BertModel, 
    AutoConfig,
    AutoTokenizer,
    AdamW,
    DataCollatorWithPadding,   
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.trainer_pt_utils import nested_detach
from utils import get_transformer_layers



class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x



class MLPParallel(nn.Module):
    def __init__(
        self, 
        input_size: int = 256,
        output_size: int = 256, 
        resid_pdrop: float = 0.5,
    ):
        super().__init__()
        self.c_fc = Conv1D(input_size, output_size).cuda(0)
        self.act = ReLUSquaredActivation().cuda(1)
        self.c_proj = Conv1D(output_size, input_size).cuda(2)
        self.dropout = nn.Dropout(resid_pdrop).cuda(3)
    
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        x = hidden_states.cuda(0)
        x = self.c_fc(x)
        x = x.cuda(1)
        x = self.act(x)
        x = x.cuda(2)
        x = self.c_proj(x)
        x = x.cuda(3)
        x = self.dropout(x)
        return x



class BertModelPipelineParallel(BertModel):
    
    def __init__(self, config, add_pooling_layer=True, devices=None):
        super().__init__(config, add_pooling_layer)
        if devices is None:
            self.num_gpus = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(self.num_gpus)]
        else:
            self.num_gpus = len(devices)
            self.devices = devices
        
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids.to(self.devices[0]) if input_ids is not None else None,
            position_ids=position_ids.to(self.devices[0]) if position_ids is not None else None,
            token_type_ids=token_type_ids.to(self.devices[0]) if token_type_ids is not None else None,
            inputs_embeds=inputs_embeds.to(self.devices[0]) if inputs_embeds is not None else None,
            past_key_values_length=past_key_values_length,
        )
        next_decoder_cache = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
        for i, layer in enumerate(self.encoder.layer):
            device_idx = i * self.num_gpus // self.config.num_hidden_layers
            device = self.devices[device_idx]
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states.to(device),
                extended_attention_mask.to(device),
                layer_head_mask.to(device) if layer_head_mask is not None else None,
                encoder_hidden_states.to(device) if encoder_hidden_states is not None else None,
                encoder_extended_attention_mask.to(device) if encoder_extended_attention_mask is not None else None,
                past_key_value.to(device) if past_key_value is not None else None,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
                    
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
            
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output.to(self.devices[-1])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



class BertForSequenceClassificationPipelineParallel(BertPreTrainedModel):
    def __init__(self, config, devices=None):
        super().__init__(config)
        if devices is None:
            self.num_gpus = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(self.num_gpus)]
        else:
            self.num_gpus = len(devices)
            self.devices = devices
        
        self.num_labels = config.num_labels
        self.config = config
        
        self.bert = BertModelPipelineParallel(config, devices=devices)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1].to(self.devices[-1])

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    

    
if __name__ == "__main__":
    import random
    import argparse
    from sklearn.datasets import fetch_20newsgroups
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--backward_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_per_steps', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='logging')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_data', type=int, default=-1)
    args = parser.parse_args()
    
    model_name = args.model_name_or_path
    batch_size = args.batch_size
    grad_accum_steps = args.gradient_accumulation_steps
    back_accum_steps = args.backward_accumulation_steps
    eval_batch_size = args.eval_batch_size
    output_dir = args.output_dir
    num_epochs = args.num_epochs
    eval_per_steps = args.eval_per_steps
    n_data = args.n_data
    seed = args.seed
    training_args = {
        'lr': 5e-5,
        'weight_decay': 0.0,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
    }
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the logging format
    logging.basicConfig(
        filename=f'{output_dir}/{model_name}_{batch_size}_{back_accum_steps}.log',
        filemode='w', # append the log file every time
        format='%(asctime)s - %(levelname)s - %(message)s', 
        level=logging.INFO,
    )
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    config = AutoConfig.from_pretrained(model_name, num_labels=20)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassificationPipelineParallel.from_pretrained(model_name, config=config, devices=None)
    logging.info(f'Putting {model.__class__.__name__} to devices {model.devices} !!!')
    
    # Assign layers to GPUs
    for i, layer in enumerate(get_transformer_layers(model)):
        device_idx = i * model.num_gpus // model.config.num_hidden_layers
        device = model.devices[device_idx]
        layer.to(device)
    model.bert.embeddings.to(model.devices[0])
    model.bert.pooler.to(model.devices[-1])
    model.dropout.to(model.devices[-1])
    model.classifier.to(model.devices[-1])
    
    # Prepare data
    def tokenize_data(examples):
        return tokenizer(examples['text'], truncation=True)
    
    collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')
    train_dataset = Dataset.from_dict({
        'text': train_data['data'],
        'label': train_data['target'],
    })
    test_dataset = Dataset.from_dict({
        'text': test_data['data'],
        'label': test_data['target'],
    })
    if n_data > 0:
        train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), n_data))
        test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), n_data))
    
    train_dataset = train_dataset.map(tokenize_data, batched=True).remove_columns(['text'])
    test_dataset = test_dataset.map(tokenize_data, batched=True).remove_columns(['text'])
    logging.info(f"Train dataset labels: {set(train_dataset['label'])}, [0]-th entry: {train_dataset[0]}")
    logging.info(f"Test dataset labels: {set(test_dataset['label'])}, [0]-th entry: {test_dataset[0]}")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        lr_lambda = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda, last_epoch)
    
    # Configure learning rate scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr = training_args['lr'],
        weight_decay = training_args['weight_decay'],
        betas = (training_args['adam_beta1'], training_args['adam_beta2']),
        eps = training_args['adam_epsilon'],
    )
    num_training_steps = len(train_dataloader) * num_epochs
    warmup_ratio = 0.1
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=math.ceil(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # Define evaluation accuracy function
    def compute_accuracy(
        model: BertForSequenceClassificationPipelineParallel, 
        dataloader: DataLoader,
    ):
        model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs = {k: v.to(model.devices[0]) if k != "labels" else v.to(model.devices[-1]) for k, v in batch.items()}
                labels = inputs['labels']
                outputs = model(**inputs)
                logits = outputs.logits
                logits = nested_detach(logits)
                if len(logits) == 1:
                    logits = logits[0]
                loss = outputs.loss.mean().detach()
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                total_samples += labels.shape[0]
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc
    
    # Training
    logging.info("Start training !!!")
    data_time, forward_time, backward_time, opt_time, eval_time = 0, 0, 0, 0, 0
    accumulated_loss = 0
    
    start = time.time()
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in pbar:
            epoch_start = time.time()
            model.train()
            optimizer.zero_grad()
            
            # Prepare inputs for model
            inputs = {k: v.to(model.devices[0]) if k != "labels" else v.to(model.devices[-1]) for k, v in batch.items()}
            data_end = time.time()
            
            # Forward pass
            outputs = model(**inputs)    
            loss = outputs.loss
            accumulated_loss += loss / grad_accum_steps
            forward_end = time.time()
            pbar.set_description(f"loss: {accumulated_loss.item():.4f}")
            
            # Backward pass
            if (step + 1) % back_accum_steps == 0:
                accumulated_loss.backward()
                accumulated_loss = 0
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), 
                #     max_norm=training_args['max_grad_norm'],
                # )
            backward_end = time.time()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                lr_scheduler.step()
            optimizer_end = time.time()
                
            if (step + 1) % eval_per_steps == 0:
                test_loss, test_acc = compute_accuracy(model, test_dataloader)
                logging.info(f"step {step+1}: test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
            
            # Accumulate time
            eval_time += time.time() - optimizer_end
            data_time += data_end - epoch_start
            forward_time += forward_end - data_end
            backward_time += backward_end - forward_end
            opt_time += optimizer_end - backward_end
        
        
    total_time = time.time() - start
    logging.info("Finished training !!!")
    logging.info("Settings: Model {} - Step {} - Back accum steps {}".format(
        model_name, step+1, back_accum_steps
    ))
    logging.info(f"Total training time: {total_time:.0f}s")
    logging.info(f"Data process time: {data_time:.0f}s ({data_time * 100 / total_time:.2f}%)")
    logging.info(f"Forward time: {forward_time:.0f}s ({forward_time * 100 / total_time:.2f}%)")
    logging.info(f"Backward time: {backward_time:.0f}s ({backward_time * 100 / total_time:.2f}%)")
    logging.info(f"Optimization time: {opt_time:.0f}s ({opt_time * 100 / total_time:.2f}%)")
    logging.info(f"Evaluation time: {eval_time:.0f}s ({eval_time * 100 / total_time:.2f}%)")

    
    
        
        
        
    
    
    
    
