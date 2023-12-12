
SETTING=random
WORKLOAD=all
BATCH_SIZE=10
N_SAMPLES=$((BATCH_SIZE * 30))
RETRAIN=0.1
OUTPUT_DIR=prof_new

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python main.py \
    --n_samples $N_SAMPLES \
    --coroutine \
    --rate_lambda 60 \
    --bptt $BATCH_SIZE \
    --nlayers 24 \
    --setting $SETTING \
    --workload $WORKLOAD \
    --use_preload \
    --output_dir $OUTPUT_DIR \
    --retraining_rate $RETRAIN


# python plot.py \
#     --node 2 \
#     --coroutine \
#     --setting $SETTING \
#     --workload $WORKLOAD \
#     --output_dir $OUTPUT_DIR \
#     --retraining_rate $RETRAIN