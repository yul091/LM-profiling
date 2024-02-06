NUM_NODES=2
NUM_SAMPLES=2000
LAYERS=12
WORKLOAD=all
SETTING=random
OUTPUT_DIR=prof_async
RETRAIN_RATE=0.2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distributed_transformer.py \
    --num_nodes $NUM_NODES \
    --n_samples $NUM_SAMPLES \
    --nlayers $LAYERS \
    --workload $WORKLOAD \
    --setting $SETTING \
    --retraining_rate $RETRAIN_RATE \
    --output_dir $OUTPUT_DIR

python plot.py \
    --node $NUM_NODES \
    --coroutine \
    --setting $SETTING \
    --workload $WORKLOAD \
    --retraining_rate $RETRAIN_RATE \
    --output_dir $OUTPUT_DIR
