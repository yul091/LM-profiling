import torch
import time
import pandas as pd
from typing import Any, Dict, Union, List
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    GPT2ForSequenceClassification,
    BartForSequenceClassification,
    DataCollatorWithPadding,
    logging,
)
logger = logging.get_logger(__name__)
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader


def extract_seq_cls_model_layers(
    model: Union[
        BertForSequenceClassification, 
        GPT2ForSequenceClassification,  
        BartForSequenceClassification,
    ]
):
    model_name = model.__class__.__name__.lower()
    layers = []
    if "bert" in model_name:
        layers = model.bert.encoder.layer
    elif "gpt2" in model_name:
        layers = model.transformer.h
    elif "bart" in model_name:
        encoder_layers = model.model.encoder.layers
        decoder_layers = model.model.decoder.layers
        layers = encoder_layers + decoder_layers
    return layers


def profile_and_infer_seq_cls(
    model: Union[
        BertForSequenceClassification, 
        GPT2ForSequenceClassification,  
        BartForSequenceClassification,
    ], 
    inputs: Dict[torch.Tensor, Any],
) -> Dict[str, float]:
    layer_times = {}
    
    # Extract the transformer layers using the provided function
    input_ids = inputs['input_ids']
    input_shape = input_ids.size()
    transformer_layers = extract_seq_cls_model_layers(model)
    batch_size, sequence_length = input_ids.shape[:2]
    if model.config.pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, model.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
            logger.warning(
                f"{model.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
    
    # For BERT and BART, there's an embedding layer before the transformer layers
    # For GPT-2, there's no such distinction
    if isinstance(model, (BertForSequenceClassification, BartForSequenceClassification)):
        embedding_layer = model.bert.embeddings if isinstance(model, BertForSequenceClassification) else model.model.encoder.embed_tokens
        start_time = time.time()
        hidden_states = embedding_layer(input_ids)
        elapsed_time = time.time() - start_time
        layer_times['embed_layer'] = elapsed_time
        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(1).unsqueeze(2)
    else:
        past_length = 0
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        inputs_embeds = model.transformer.wte(input_ids)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
    
    # Profile each transformer layer
    for i, layer in enumerate(transformer_layers):
        start_time = time.time()
        if isinstance(model, GPT2ForSequenceClassification):
            outputs = layer(hidden_states)   
        elif isinstance(model, BartForSequenceClassification):
            # Expand the mask to have shape (batch_size, 1, sequence_length, sequence_length)
            attention_mask = inputs["attention_mask"].squeeze(1).squeeze(1)  # Ensure it has shape (batch_size, sequence_length)
            expanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, sequence_length, sequence_length)
            # For BART, we pass in both the hidden states and attention mask, and layer_head_mask as None
            outputs = layer(hidden_states, expanded_attention_mask, None)
        else:
            outputs = layer(hidden_states, inputs["attention_mask"])
        hidden_states = outputs[0]
        elapsed_time = time.time() - start_time
        layer_times[f'transformer_layer_{i}'] = elapsed_time
    
    # Get the classification output from the model
    if isinstance(model, GPT2ForSequenceClassification):
        logits = model.score(hidden_states)
    elif isinstance(model, BartForSequenceClassification):
        logits = model.classification_head(hidden_states)
    else:
        logits = model.classifier(hidden_states)
    
    # Decode the output to get the predicted class
    pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    predictions = torch.argmax(pooled_logits, dim=-1)
    decoded_predictions = [model.config.id2label[prediction.item()] for prediction in predictions]

    return layer_times, decoded_predictions


def text_classification_profiling(args: argparse.Namespace):
    model_name_or_path = args.model_name_or_path
    per_device_eval_batch_size = args.per_device_eval_batch_size
    output_dir = args.output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gpt' in model_name_or_path.lower():
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load imdb and stanford sentiment treebank (sst-2) / movie review (mr) datasets from Huggingface
    imdb = load_dataset('imdb') # train (25000)/test (25000)/unsupervised (50000): ['text', 'label']
    train_dataset = imdb['train']
    test_dataset = imdb['test']

    num_labels = train_dataset.features['label'].num_classes
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, 
        config=config, 
        ignore_mismatched_sizes=True,
    )
    if 'gpt' in model_name_or_path.lower():
        # Resize only if it hasn't been resized already
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        # Set the padding token in the model's configuration
        config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    # Tokenize the datasets
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    # Remove unused columns (keep ['label', 'input_ids', 'token_type_ids', 'attention_mask'])
    tokenized_train = tokenized_train.remove_columns(['text'])
    tokenized_test = tokenized_test.remove_columns(['text'])
    # Set the format of the datasets to PyTorch tensors
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Batch inference with the model
    test_loader = DataLoader(
        tokenized_test, 
        batch_size=per_device_eval_batch_size, 
        shuffle=False,
        collate_fn=data_collator,
    )
    model.eval()
    model.to('cuda')
    profiling_res = []

    for i, batch in tqdm(enumerate(test_loader)):
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        layer_times, predictions = profile_and_infer_seq_cls(model, inputs)
        profiling_res.append(layer_times)
        # print("predictions:", predictions)
        # print("layer_times:", layer_times)


    profiling_df = pd.DataFrame(profiling_res)
    model_n = model_name_or_path.split('/')[-1]
    profiling_df.to_csv(f'{output_dir}/profiling_{model_n}_res.csv', index=False)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    argparse.add_argument('--per_device_eval_batch_size', type=int, default=5)
    argparse.add_argument('--output_dir', type=str, default='profile_res')
    args = argparse.parse_args()
    
    text_classification_profiling(args)