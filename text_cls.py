import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Optional
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Value
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from layer_profiler import LayerProfiler



class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


class SequenceClassificationProflier:
    
    def __init__(
        self,
        args: argparse.Namespace, 
        logger: Optional[logging.Logger] = None
    ):
        self.args = args
        self.logger = logger
        model_name_or_path = args.model_name_or_path
        per_device_eval_batch_size = args.per_device_eval_batch_size
        output_dir = args.output_dir
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if 'gpt' in model_name_or_path.lower():
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        # Load imdb and stanford sentiment treebank (sst-2) / movie review (mr) datasets from Huggingface
        raw_datasets = load_dataset(args.dataset) # train (25000)/test (25000)/unsupervised (50000): ['text', 'label']
        self.train_dataset = imdb['train']
        self.test_dataset = imdb['test']

        num_labels = self.train_dataset.features['label'].num_classes
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, 
            num_labels=num_labels,
            finetuning_task="text-classification",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            config=self.config, 
            ignore_mismatched_sizes=True,
        )
        if 'gpt' in model_name_or_path.lower():
            # Resize only if it hasn't been resized already
            if self.model.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
            # Set the padding token in the model's configuration
            self.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        is_regression = (
            raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
            if args.do_regression is None
            else args.do_regression
        )
        is_multi_label = False
         
        if is_regression:
            label_list = None
            num_labels = 1
            # regession requires float as label type, let's cast it if needed
            for split in raw_datasets.keys():
                if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                    logger.warning(
                        f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                    )
                    features = raw_datasets[split].features
                    features.update({"label": Value("float32")})
                    try:
                        raw_datasets[split] = raw_datasets[split].cast(features)
                    except TypeError as error:
                        logger.error(
                            f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                        )
                        raise error

        else:  # classification
            if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
                is_multi_label = True
                logger.info("Label type is list, doing multi-label classification")
            # Trying to find the number of labels in a multi-label classification task
            # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
            # So we build the label list from the union of labels in train/val/test.
            label_list = get_label_list(raw_datasets, split="train")
            for split in ["validation", "test"]:
                if split in raw_datasets:
                    val_or_test_labels = get_label_list(raw_datasets, split=split)
                    diff = set(val_or_test_labels).difference(set(label_list))
                    if len(diff) > 0:
                        # add the labels that appear in val/test but not in train, throw a warning
                        logger.warning(
                            f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                        )
                        label_list += list(diff)
            # if label is -1, we throw a warning and remove it from the label list
            for label in label_list:
                if label == -1:
                    logger.warning("Label -1 found in label list, removing it.")
                    label_list.remove(label)
            
        def preprocess_function(examples):
            return self.tokenizer(examples['text'], truncation=True)
        
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if is_regression:
                preds = np.squeeze(preds)
                result = metric.compute(predictions=preds, references=p.label_ids)
            elif is_multi_label:
                preds = np.array([np.where(p > 0.5, 1, 0) for p in preds])
                # Micro F1 is commonly used in multi-label classification
                result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
            else:
                preds = np.argmax(preds, axis=1)
                result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

        # Tokenize the datasets
        self.train_dataset = self.train_dataset.map(preprocess_function, batched=True)
        self.test_dataset = self.test_dataset.map(preprocess_function, batched=True)
        # Remove unused columns (keep ['label', 'input_ids', 'token_type_ids', 'attention_mask'])
        self.train_dataset = self.train_dataset.remove_columns(['text'])
        self.test_dataset = self.test_dataset.remove_columns(['text'])
        
        # Set the format of the datasets to PyTorch tensors
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
        # Profiling
        self.profiler = LayerProfiler(self.model, self.config, logger=logger)


def inference():
    
    
    

    
        
    

    
    

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
        
        
    layer_time_forward = profiler.layer_time_forward
    layer_input_length = profiler.layer_input_length
    layer_memory = profiler.layer_memory_forward

    # Save the results
    model_n = model_name_or_path.split('/')[-1]
    with open(f'{output_dir}/latency_forward_{model_n}.json', 'w') as f:
        json.dump(layer_time_forward, f)
    with open(f'{output_dir}/memory_forward_{model_n}.json', 'w') as f:
        json.dump(layer_memory, f)
    with open(f'{output_dir}/input_length_{model_n}.json', 'w') as f:
        json.dump(layer_input_length, f)


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb')
    parser.add_argument('--do_regression', type=bool, default=None)
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