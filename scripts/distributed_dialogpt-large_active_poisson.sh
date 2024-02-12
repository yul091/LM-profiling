NUM_NODES=2
NUM_SAMPLES=600
BATCH_SIZE=3
WORKLOAD=poisson
SETTING=active
OUTPUT_DIR=prof_async
MODEL_NAME=dialogpt-large

for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distributed_dialogpt.py \
        --model_name_or_path "microsoft/DialoGPT-large" \
        --model_name $MODEL_NAME \
        --num_nodes $NUM_NODES \
        --n_samples $NUM_SAMPLES \
        --workload $WORKLOAD \
        --setting $SETTING \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE

    python plot.py \
        --node $NUM_NODES \
        --model_name $MODEL_NAME \
        --setting $SETTING \
        --workload $WORKLOAD \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR
done