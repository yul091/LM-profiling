import os
import sys
import time
import math
import inspect
import shutil
# import wandb
import logging
from typing import Dict, Union, Any, List, Tuple, Optional, Callable
from collections.abc import Mapping
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import hp_params

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from packaging import version
from transformers import (
    BertModel,
    BertForSequenceClassification,
    GPT2Model,
    GPT2ForSequenceClassification,
    XLNetModel,
    XLNetForSequenceClassification,
    BartModel,
    BartForSequenceClassification,
    Trainer,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
    is_sagemaker_mp_enabled,
    is_apex_available,
    is_torch_tpu_available,
    is_accelerate_available,
    is_peft_available,
    is_datasets_available,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    EvalPrediction,
    has_length,
    speed_metrics,
    seed_worker,
)
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler
from transformers.trainer_callback import TrainerState, TrainerCallback
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, 
    BertModel,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
    # _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    _CONFIG_FOR_DOC,
    _SEQ_CLASS_EXPECTED_OUTPUT,
    _SEQ_CLASS_EXPECTED_LOSS,
)
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

if is_datasets_available():
    import datasets

if is_apex_available():
    from apex import amp
    
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
    
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    
if is_peft_available():
    from peft import PeftModel

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

   
COLOR_MAP = {
    'embedding': 'blue',
    'attention': 'purple',
    'ffn': 'brown',
    'dropout': 'grey',
    'backward': 'green',
}


def get_transformer_layers(
    model: Union[
        BertForSequenceClassification, 
        GPT2ForSequenceClassification, 
        XLNetForSequenceClassification, 
        BartForSequenceClassification,
    ]
) -> List[Union[BertModel, GPT2Model, XLNetModel, BartModel]]:
    model_name = model.__class__.__name__.lower()
    layers = []
    if "bert" in model_name:
        layers = model.bert.encoder.layer
    elif "gpt2" in model_name:
        layers = model.transformer.h
    elif "xlnet" in model_name:
        layers = model.transformer.layer
    elif "bart" in model_name:
        encoder_layers = model.model.encoder.layers
        decoder_layers = model.model.decoder.layers
        layers = encoder_layers + decoder_layers
    return layers

def get_colors(index: List[str], color_map: dict = COLOR_MAP):
    return [
        color_map['embedding'] if 'embedding' in idx.lower() 
        else color_map['attention'] if 'attention' in idx.lower() or 'attn' in idx.lower()
        else color_map['layernorm'] if 'ln' in idx.lower() or 'layernorm' in idx.lower()
        else color_map['ffn'] if (
            'mlp' in idx.lower() or 
            'linear' in idx.lower() or 
            'pooler' in idx.lower() or 
            'intermediate' in idx.lower() or
            'output' in idx.lower()
        )
        else color_map['dropout'] if 'dropout' in idx.lower()
        else color_map['backward'] if 'backward' in idx.lower() or 'bp' in idx.lower()
        else 'red'  # default color 
        for idx in index]

 

# Plot the average latency distribution of each layer
def plot_layer_profiling(
    profile_res: pd.DataFrame, 
    model_name: str, 
    backward_res: pd.DataFrame = None,
    save_file: str = None,
    color_map: dict = COLOR_MAP,
    metric: str = 'inference latency',
    unit: str = 'seconds',
    figsize: Tuple[int, int] = (20, 6),
):
    # Assuming you have the DataFrame loaded as df (do not include the batch_size, input_length columns)
    if 'batch_size' in profile_res.columns and 'input_length' in profile_res.columns:
        res = profile_res.drop(columns=['batch_size', 'input_length'])
    else:
        res = profile_res
    averages = res.mean()
    
    # Determine the color of each bar based on its label
    colors = get_colors(averages.index)
    
    # Create custom patches for legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[key], label=key) for key in color_map]

    # Plotting
    plt.figure(figsize=figsize)
    averages.plot(kind='bar', color=colors, width=0.5)
    
    # Also plot line graph
    plt.plot(averages, color='black', linestyle='-', linewidth=2)
    
    plt.ylabel(f'Average {metric} ({unit})', fontdict={'fontsize': 12})
    plt.xlabel('Layer', fontdict={'fontsize': 12})
    plt.title(f'Average {metric} per Layer for {model_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Add legend for the 6 layers
    plt.legend(handles=legend_elements, title="Layer type")
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show() 
    
    

def plot_layer_profiling_dist(
    profile_res: pd.DataFrame, 
    model_name: str, 
    save_file: str = None,
    color_map: dict = COLOR_MAP,
    metric: str = 'inference latency',
    unit: str = 'seconds',
    figsize: Tuple[int, int] = (20, 6),
):
    
    # Assuming you have the DataFrame loaded as df (do not include the batch_size, input_length columns)
    # If res has columns batch_size and input_length, drop them
    if 'batch_size' in profile_res.columns and 'input_length' in profile_res.columns:
        res = profile_res.drop(columns=['batch_size', 'input_length'])
    else:
        res = profile_res
    
    # Determine the color of each column based on its label
    column_colors = get_colors(res.columns)

    # Create custom patches for legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[key], label=key) for key in color_map]
    
    # Plotting
    plt.figure(figsize=figsize)
    
    # Boxplot
    boxprops = dict(linestyle='-', linewidth=1)
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    # res.boxplot(column=res.columns, vert=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
    bp = res.boxplot(
        vert=True, 
        patch_artist=True, 
        boxprops=boxprops, 
        medianprops=medianprops, 
        showfliers=False, 
        return_type='dict',
    )
    
    # Coloring the boxes based on the determined colors
    for patch, color in zip(bp['boxes'], column_colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Layer', fontdict={'fontsize': 12})
    plt.ylabel(f'{metric} ({unit})', fontdict={'fontsize': 12})
    plt.title(f'Distribution of {metric} per Layer for {model_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Add legend for the layer types with a title
    plt.legend(handles=legend_elements, title="Layer type")
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show()
    



# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
    
        
        
# @dataclass
# class ActiveSelectionLabelSmoother:
#     """
#     Adds label-smoothing on a pre-computed output from a Transformers model.

#     Args:
#         epsilon (`float`, *optional*, defaults to 0.1):
#             The label smoothing factor.
#         ignore_index (`int`, *optional*, defaults to -100):
#             The index in the labels to ignore when computing the loss.
#     """

#     epsilon: float = 0.1
#     ignore_index: int = -100

#     def __call__(self, model_output, labels, shift_labels=False, indices=None):
#         logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
#         if indices is not None:
#             logits = logits[indices]
#             labels = labels[indices]
        
#         if shift_labels:
#             logits = logits[..., :-1, :].contiguous()
#             labels = labels[..., 1:].contiguous()

#         log_probs = -nn.functional.log_softmax(logits, dim=-1)
#         if labels.dim() == log_probs.dim() - 1:
#             labels = labels.unsqueeze(-1)

#         padding_mask = labels.eq(self.ignore_index)
#         # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
#         # will ignore them in any case.
#         labels = torch.clamp(labels, min=0)
#         nll_loss = log_probs.gather(dim=-1, index=labels)
#         # works for fp16 input tensor too, by internally upcasting it to fp32
#         smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

#         nll_loss.masked_fill_(padding_mask, 0.0)
#         smoothed_loss.masked_fill_(padding_mask, 0.0)

#         # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
#         num_active_elements = padding_mask.numel() - padding_mask.long().sum()
#         nll_loss = nll_loss.sum() / num_active_elements
#         smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
#         return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
