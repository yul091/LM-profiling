NUM_NODES=2
NUM_SAMPLES=300
BATCH_SIZE=3
LAYERS=12
WORKLOAD=poisson
SETTING=active
OUTPUT_DIR=prof_async
RETRAIN_RATE=0

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
