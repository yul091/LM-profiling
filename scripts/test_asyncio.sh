NUM_NODES=2
NUM_SAMPLES=200
WORKLOAD=all
SETTING=random
OUTPUT_DIR=prof_async

CUDA_VISIBLE_DEVICES=0,1,4,5 python test_asyncio.py \
    --num_nodes $NUM_NODES \
    --n_samples $NUM_SAMPLES \
    --workload $WORKLOAD \
    --setting $SETTING \
    --output_dir $OUTPUT_DIR

python plot.py \
    --node $NUM_NODES \
    --coroutine \
    --setting $SETTING \
    --workload $WORKLOAD \
    --output_dir $OUTPUT_DIR
