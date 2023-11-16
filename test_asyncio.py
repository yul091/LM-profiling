import os
import sys
sys.dont_write_bytecode = True
import json
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from dataset import get_data, SentencePairDataset
from models import Encoder, Decoder, PipelineStage
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Subset
from utils import record_time, get_total_params
from collections import defaultdict


# Dummy stage functions for illustration purposes
def stage1_inference(data, stage, timing_info):
    torch.cuda.set_device(0)
    # Your inference code for stage 1 on GPU 0 goes here
    print(f"Running stage 1 on GPU {torch.cuda.current_device()}")
    # Simulate some work
    # torch.cuda._sleep(5000000000)
    record_time(0, 'start', 'forward', timing_info)
    output = stage(data)
    record_time(0, 'end', 'forward', timing_info)
    return 'output 1'

def stage2_inference(data, stage, timing_info):
    torch.cuda.set_device(1)
    # Your inference code for stage 2 on GPU 1 goes here
    print(f"Running stage 2 on GPU {torch.cuda.current_device()}")
    # Simulate some work
    # torch.cuda._sleep(5000000000)
    record_time(1, 'start', 'forward', timing_info)
    output = stage(data)
    record_time(1, 'end', 'forward', timing_info)
    return 'output 2'

# Main function to run both stages concurrently
def run_stages_concurrently(data1, data2, stage1, stage2, timing_info):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(stage1_inference, data1, stage1, timing_info)
        future2 = executor.submit(stage2_inference, data2, stage2, timing_info)
        
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
num_gpus = 2
emsize = 4096
nhead = 8
nhid = 4096
dropout = 0.2

stage1 = nn.Sequential(
    Encoder(ntokens, emsize, dropout),
    *[TransformerEncoderLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers)],
    Decoder(ntokens, emsize),
).cuda(0)

stage2 = nn.Sequential(
    Encoder(ntokens, emsize, dropout),
    *[TransformerEncoderLayer(emsize, nhead, nhid, dropout) for _ in range(nlayers)],
    Decoder(ntokens, emsize),
).cuda(1)

data_for_stage1 = next(iter(test_loader))[0].cuda(0)
data_for_stage2 = next(iter(test_loader))[0].cuda(1)

timing_info = defaultdict(list)

# Run the stages concurrently
run_stages_concurrently(data_for_stage1, data_for_stage2, stage1, stage2, timing_info)
output_dir = 'prof'

os.makedirs(output_dir, exist_ok=True)
stats_f = f'{output_dir}/test_asyncio.json'
with open(stats_f, 'w') as f:
    json.dump(timing_info, f, indent=4)
