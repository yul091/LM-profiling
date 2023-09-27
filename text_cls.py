import sys
sys.dont_write_bytecode = True
import pandas as pd
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    logging,
)
logger = logging.get_logger(__name__)

from profile_gpt2 import GPT2ForSequenceClassificationProfile





def text_classification_profiling(args: argparse.Namespace):
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

    if 'gpt' in model_name_or_path.lower():
        model = GPT2ForSequenceClassificationProfile.from_pretrained(
            model_name_or_path, 
            config=config, 
            ignore_mismatched_sizes=True,
        )
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
    model.to(device)
    latency_res = []
    memory_res = []

    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Get input length
        batch_size, sequence_length = batch['input_ids'].shape[:2]
        layer_times = {'batch_size': batch_size, 'input_length': sequence_length}
        layer_memory = {'batch_size': batch_size, 'input_length': sequence_length}
        
        # Prepare inputs
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        
        outputs = model(**inputs, layer_times=layer_times, layer_memory=layer_memory)
        pooled_logits = outputs[1]
        
        # Decode the output to get the predicted class
        predictions = torch.argmax(pooled_logits, dim=-1)
        decoded_predictions = [model.config.id2label[prediction.item()] for prediction in predictions]
    
        latency_res.append(layer_times)
        memory_res.append(layer_memory)

    latency_df = pd.DataFrame(latency_res)
    memory_df = pd.DataFrame(memory_res)
    model_n = model_name_or_path.split('/')[-1]
    latency_df.to_csv(f'{output_dir}/latency_{model_n}_res.csv', index=False)
    memory_df.to_csv(f'{output_dir}/memory_{model_n}_res.csv', index=False)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    argparse.add_argument('--per_device_eval_batch_size', type=int, default=5)
    argparse.add_argument('--output_dir', type=str, default='profile_res')
    args = argparse.parse_args()
    
    text_classification_profiling(args)