
import time
import math
from functools import partial
from collections import defaultdict
from typing import Callable, List, Union, Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
from torch.nn.modules.module import Module, _grad_t
from torch.utils.hooks import RemovableHandle


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)
    

class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)
    
    
class PipelineStage(nn.Module):
    def __init__(self, layers: List[TransformerEncoderLayer], device: int, timing_info: dict = None):
        super(PipelineStage, self).__init__()
        self.layers = nn.Sequential(*layers).cuda(device)
        self.device = device
        self.register_full_backward_hook(partial(self.backward_hook, timing_info=timing_info))

    def forward(self, x):
        return self.layers(x)
    
    # Add a method to profile the backward pass
    def backward_hook(
        self, 
        module: nn.Module, 
        grad_input: Tuple[torch.Tensor], 
        grad_output: Tuple[torch.Tensor], 
        timing_info: dict = None,
    ):
        # print(f"Backward pass started for {module}")
        start_time = time.time()
        if f"{self.device+1}_start" in timing_info:
            timing_info[f"{self.device+1}_end"].append((start_time, "backward"))
        timing_info[f"{self.device}_start"].append((start_time, "backward"))
        
        
def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

    
def get_stages(
    ntokens: int, 
    nlayers: int, 
    num_gpus: int, 
    emsize: int, 
    nhead: int, 
    nhid: int, 
    dropout: float, 
    init_device: int = 0,
    timing_info: dict = None,
) -> List[PipelineStage]:
    # Create pipeline stages
    partition_len = ((nlayers - 1) // num_gpus) + 1
    # Add encoder in the beginning.
    tmp_list = [Encoder(ntokens, emsize, dropout)]
    stages = []
    # Add all the necessary transformer blocks
    for i in range(nlayers):
        transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        if i != 0 and i % (partition_len) == 0:
            # Create a new pipeline stage
            stage_device = i // (partition_len) - 1 + init_device
            print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device))
            stages.append(PipelineStage(tmp_list, stage_device, timing_info))
            tmp_list = []
        tmp_list.append(transformer_block)
        
    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize))
    stages.append(PipelineStage(tmp_list, stage_device + 1, timing_info))
    print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device + 1))
    print ('Total parameters in model: {:,}'.format(get_total_params(torch.nn.Sequential(*stages))))
    return stages


def stages_forward(
    stages: List[PipelineStage], 
    inputs: Tensor, 
    # timing_info: dict, 
    # init_device: int = 0,
):
    # Forward pass
    hidden = inputs.clone()
    for i, stage in enumerate(stages):
        # record_time(device, 'start', 'forward', timing_info)
        hidden = stage(hidden.cuda(stage.device))
        # record_time(device, 'end', 'forward', timing_info)
    return hidden
            
        
        
if __name__ == '__main__':
    torch.manual_seed(0) # set random seed
    model_kwargs = {
        'nlayers': 12,
        'emsize': 1024,
        'nhead': 8,
        'nhid': 1024,
        'dropout': 0.2,
        'ntokens': 10000,
    }
    num_gpus_per_node = 4
    timing_info = defaultdict(list)
    stages = get_stages(
        num_gpus=num_gpus_per_node,
        init_device=0,
        timing_info=timing_info,
        **model_kwargs,
    )
    inputs = torch.randint(0, model_kwargs['ntokens'], (100, 32)).cuda(0) # (B X T)
    outputs = stages_forward(stages, inputs) # (B X T X C)
    output_flat = outputs.contiguous().view(-1, model_kwargs['ntokens']) # (B * T, C)
    labels = inputs.contiguous().view(-1).cuda(3) # (B * T)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output_flat, labels)
    
    # Backward pass (integration test)
    loss.backward()
    timing_info['0_end'].append((time.time(), 'backward'))
    print(loss)
    print(stages[-1].layers[-1].decoder.weight.grad)
    print(stages[0].layers[0].encoder.weight.grad)
    print(timing_info)