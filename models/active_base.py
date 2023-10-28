from typing import Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, SequenceClassifierOutput


class ActiveSelectionBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, *args) -> Union[Tuple, SequenceClassifierOutput, SequenceClassifierOutputWithPast]:
        raise NotImplementedError
        
    def _get_loss(self, *args) -> torch.Tensor:
        raise NotImplementedError