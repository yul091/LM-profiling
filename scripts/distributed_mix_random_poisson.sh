
SETTING=random
WORKLOAD=poisson
BATCH_SIZE=2
N_SAMPLES=$((BATCH_SIZE * 30))

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py \
    --n_samples $N_SAMPLES \
    --coroutine \
    --rate_lambda 60 \
    --bptt $BATCH_SIZE \
    --nlayers 24 \
    --setting $SETTING \
    --workload $WORKLOAD


python plot.py \
    --node 2 \
    --coroutine \
    --setting $SETTING \
    --workload $WORKLOAD