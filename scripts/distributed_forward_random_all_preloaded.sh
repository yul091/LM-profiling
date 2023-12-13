
SETTING=random
WORKLOAD=all
BATCH_SIZE=10
N_SAMPLES=$((BATCH_SIZE * 30))
RETRAIN=0
NUM_NODES=2

CUDA_VISIBLE_DEVICES=0,1,2,5,6,7 python distributed_system.py \
    --n_samples $N_SAMPLES \
    --rate_lambda 60 \
    --bptt $BATCH_SIZE \
    --nlayers 24 \
    --setting $SETTING \
    --workload $WORKLOAD \
    --use_preload \
    --num_nodes $NUM_NODES \
    --retraining_rate $RETRAIN


python plot.py \
    --node $NUM_NODES \
    --coroutine \
    --setting $SETTING \
    --workload $WORKLOAD \
    --retraining_rate $RETRAIN