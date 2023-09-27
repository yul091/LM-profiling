import sys
sys.dont_write_bytecode = True
import json
import pandas as pd
import argparse
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# from profile_gpt2 import GPT2ForSequenceClassificationProfile
from layer_profiler import LayerProfiler



def inference(args: argparse.Namespace, logger: logging.Logger = None):
    model_name_or_path = args.model_name_or_path
    per_device_eval_batch_size = args.per_device_eval_batch_size
    output_dir = args.output_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        
    # Profiling
    profiler = LayerProfiler(model, config, logger=logger)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    # Tokenize the datasets
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True).select(range(1000)) # Only use 1000 samples for testing
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
    model.to(device)

    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Prepare inputs
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        # outputs = model(**inputs, layer_times=layer_times, layer_memory=layer_memory)
        try:
            outputs = model(**inputs)
        except:
            inputs['decoder_input_ids'] = inputs['input_ids']
            inputs['decoder_attention_mask'] = inputs['attention_mask']
            outputs = model(**inputs)
        
        pooled_logits = outputs[1]
        
        # Decode the output to get the predicted class
        predictions = torch.argmax(pooled_logits, dim=-1)
        decoded_predictions = [model.config.id2label[prediction.item()] for prediction in predictions]
        # logger.info(f"Predictions: {decoded_predictions}")
        
        
    layer_time_cost = profiler.layer_time_cost
    layer_input_length = profiler.layer_input_length
    layer_memory = profiler.layer_memory

    # Save the results
    model_n = model_name_or_path.split('/')[-1]
    with open(f'{output_dir}/latency_{model_n}_res.json', 'w') as f:
        json.dump(layer_time_cost, f)
    with open(f'{output_dir}/memory_{model_n}_res.json', 'w') as f:
        json.dump(layer_memory, f)
    with open(f'{output_dir}/input_length_{model_n}_res.json', 'w') as f:
        json.dump(layer_input_length, f)


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='profile_res')
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
    
    inference(args, logger=logger)