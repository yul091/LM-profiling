NUM_NODES=2

CUDA_VISIBLE_DEVICES=0,1,4,5 python test_asyncio.py --num_nodes $NUM_NODES --workload all --setting random

python plot.py --test_asyncio