
SETTING=variant
WORKLOAD=poisson
BATCH_SIZE=10
N_SAMPLES=$((BATCH_SIZE * 30))
RETRAIN=0

# CUDA_VISIBLE_DEVICES=0,1,2,5,6,7 /usr/lib/nsight-systems/bin/nsys profile \
#     -f true \
#     -c cudaProfilerApi \
#     -o prof/nsys_variant_poisson \
CUDA_VISIBLE_DEVICES=0,1,2,5,6,7 python main.py \
    --n_samples $N_SAMPLES \
    --coroutine \
    --rate_lambda 60 \
    --bptt $BATCH_SIZE \
    --nlayers 24 \
    --setting $SETTING \
    --workload $WORKLOAD \
    --use_preload \
    --retraining_rate $RETRAIN


python plot.py \
    --node 2 \
    --coroutine \
    --setting $SETTING \
    --workload $WORKLOAD \
    --retraining_rate $RETRAIN