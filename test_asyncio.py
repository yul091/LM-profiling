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


def node_inference(
        data: torch.Tensor,
        stages: List[PipelineStage], 
        device: int, 
        timing_info: dict,
    ):
        # torch.cuda.set_device(device)  # Set the current device to the stage's GPU
        hidden = data.clone()
        for i, stage in enumerate(stages):
            record_time(device+i, 'start', 'forward', timing_info)
            hidden = stage(hidden)
            record_time(device+i, 'end', 'forward', timing_info)
            if i != len(stages) - 1:
                # Need to send the output to the next stage, except for the last stage
                hidden = hidden.cuda(device+i+1, non_blocking=True) 
        return hidden


# Main function to run both stages concurrently
def run_stages_concurrently(data1, data2, stages1, stages2, timing_info, device1=0, device2=1):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(node_inference, data1, stages1, device1, timing_info)
        future2 = executor.submit(node_inference, data2, stages2, device2, timing_info)
        
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
num_gpus = torch.cuda.device_count() // 2
emsize = 4096
nhead = 8
nhid = 4096
dropout = 0.2
initial_device = 0

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


stage1 = get_stages(nlayers, num_gpus, emsize, nhead, nhid, dropout, init_device=initial_device)
stage2 = get_stages(nlayers, num_gpus, emsize, nhead, nhid, dropout, init_device=initial_device+num_gpus)

data_for_stage1 = next(iter(test_loader))[0].cuda(initial_device)
data_for_stage2 = next(iter(test_loader))[0].cuda(initial_device+num_gpus)

timing_info = defaultdict(list)

# Run the stages concurrently
run_stages_concurrently(
    data_for_stage1, 
    data_for_stage2, 
    stage1, 
    stage2, 
    timing_info, 
    device1=initial_device, 
    device2=initial_device+num_gpus,
)
output_dir = 'prof'
print("Timing info: ", timing_info)

os.makedirs(output_dir, exist_ok=True)
stats_f = f'{output_dir}/test_asyncio.json'
with open(stats_f, 'w') as f:
    json.dump(timing_info, f, indent=4)
