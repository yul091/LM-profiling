import os
import sys
sys.dont_write_bytecode = True
import json
import torch
import torch.nn as nn
# import asyncio
from typing import List
from torch.nn import TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from dataset import get_data, SentencePairDataset
from models import Encoder, Decoder, PipelineStage
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Subset
from utils import record_time, get_total_params
from collections import defaultdict


def stage_inference(
        stages: List[PipelineStage], 
        device: int, 
        data: torch.Tensor,
        timing_info: dict,
    ):
        # torch.cuda.set_device(device)  # Set the current device to the stage's GPU
        hidden = data.clone()
        for i, stage in enumerate(stages):
            record_time(device+i, 'start', 'forward', timing_info)
            hidden = stage(hidden)
            record_time(device+i, 'end', 'forward', timing_info)
            hidden = hidden.cuda(device+i+1, non_blocking=True) 


# Dummy stage functions for illustration purposes
def stage1_inference(data, stages, timing_info, device):
    torch.cuda.set_device(device)
    # Your inference code for stage 1 on GPU 0 goes here
    print(f"Running stage 1 on GPU {torch.cuda.current_device()}")
    # Simulate some work
    stage_inference(stages, device, data, timing_info)
    return 'output 1'

def stage2_inference(data, stages, timing_info, device):
    torch.cuda.set_device(device)
    # Your inference code for stage 2 on GPU 1 goes here
    print(f"Running stage 2 on GPU {torch.cuda.current_device()}")
    # Simulate some work
    # torch.cuda._sleep(5000000000)
    stage_inference(stages, device, data, timing_info)
    return 'output 2'

# Main function to run both stages concurrently
def run_stages_concurrently(data1, data2, stages1, stages2, timing_info, device1=0, device2=1):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(stage1_inference, data1, stages1, timing_info, device1)
        future2 = executor.submit(stage2_inference, data2, stages2, timing_info, device2)
        
        # Wait for both stages to complete
        result1 = future1.result()
        result2 = future2.result()
        print(result1, result2)

# Example data for each stage
_, _, test_data, vocab = get_data(setting='identical', eval_batch_size=128)
def collate_batch(batch):
    # 'batch' is a list of tuples with (sequence, target)
    batch_data, batch_target = zip(*batch)
    combined_list = batch_data + batch_target
    # Dynamically pad the batch
    padded = pad_sequence(combined_list, batch_first=True, padding_value=vocab['<pad>'])
    padded_data = padded[:len(batch_data)]
    padded_target = padded[len(batch_data):]
    return padded_data, padded_target.view(-1)

test_dataset = SentencePairDataset(test_data, setting='identical')
ntokens = len(vocab) # the size of vocabulary
test_loader = DataLoader(
    test_dataset, 
    batch_size=25, 
    collate_fn=collate_batch,
    shuffle=False,
)
nlayers = 24
num_gpus = 3
emsize = 4096
nhead = 8
nhid = 4096
dropout = 0.2

def get_stages(nlayers, num_gpus, emsize, nhead, nhid, dropout, init_device=0):
    # Create pipeline stages
    partition_len = ((nlayers - 1) // num_gpus) + 1
    # Add encoder in the beginning.
    tmp_list = [Encoder(ntokens, emsize, dropout)]
    stages = []

    # Add all the necessary transformer blocks.
    for i in range(nlayers):
        transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        if i != 0 and i % (partition_len) == 0:
            # Create a new pipeline stage
            stage_device = i // (partition_len) - 1 + init_device
            print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device))
            stages.append(PipelineStage(tmp_list, stage_device))
            tmp_list = []
            
        tmp_list.append(transformer_block)
        
    # Add decoder in the end.
    tmp_list.append(Decoder(ntokens, emsize))
    stages.append(PipelineStage(tmp_list, stage_device + 1))
    print("Put stage {} on device {}".format([layer.__class__.__name__ for layer in tmp_list], stage_device + 1))
    print ('Total parameters in model: {:,}'.format(get_total_params(torch.nn.Sequential(*stages))))
    return stages


stage1 = get_stages(nlayers, num_gpus, emsize, nhead, nhid, dropout, init_device=0)
stage2 = get_stages(nlayers, num_gpus, emsize, nhead, nhid, dropout, init_device=num_gpus)

data_for_stage1 = next(iter(test_loader))[0].cuda(0)
data_for_stage2 = next(iter(test_loader))[0].cuda(num_gpus)

timing_info = defaultdict(list)

# Run the stages concurrently
run_stages_concurrently(data_for_stage1, data_for_stage2, stage1, stage2, timing_info, device1=0, device2=num_gpus)
output_dir = 'prof'
print("Timing info: ", timing_info)

os.makedirs(output_dir, exist_ok=True)
stats_f = f'{output_dir}/test_asyncio.json'
with open(stats_f, 'w') as f:
    json.dump(timing_info, f, indent=4)
