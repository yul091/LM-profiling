NUM_NODES=2
NUM_SAMPLES=600
BATCH_SIZE=3
WORKLOAD=poisson
SETTING=active
OUTPUT_DIR=prof_async
# RETRAIN_RATE=0.8

for RETRAIN_RATE in 0.8 0.9 1.0; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distributed_dialogpt.py \
        --model_name_or_path "microsoft/DialoGPT-small" \
        --num_nodes $NUM_NODES \
        --n_samples $NUM_SAMPLES \
        --workload $WORKLOAD \
        --setting $SETTING \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE

    python plot.py \
        --node $NUM_NODES \
        --model_name "dialogpt" \
        --setting $SETTING \
        --workload $WORKLOAD \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR
done