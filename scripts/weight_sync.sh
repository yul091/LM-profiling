NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.5
SETTING=isolated
WORKLOAD=poisson
BASE_DIR=prof_weight_sync
EXPERIMENTS=1 # each experiment run 3 times
RATE_LAMBDA=5
RETRAIN_RATE=0.5

# NUM_NODES=2
# for MODEL_NAME in "DialoGPT-large"; do
#     for SETTING in isolated; do
#         OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/${MODEL_NAME}
#         python distributed_dialogpt.py \
#             --model_name_or_path "microsoft/$MODEL_NAME" \
#             --model_name $MODEL_NAME \
#             --num_nodes $NUM_NODES \
#             --num_gpus $NUM_NODES \
#             --n_samples $NUM_SAMPLES \
#             --rate_lambda $RATE_LAMBDA \
#             --workload $WORKLOAD \
#             --setting $SETTING \
#             --lr 2e-6 \
#             --retraining_rate $RETRAIN_RATE \
#             --task_assignment 'rr' \
#             --output_dir $OUTPUT_DIR \
#             --batch_size $BATCH_SIZE \
#             --experiments $EXPERIMENTS \
#             --memory_threshold $MEMORY_THRESHOLD

#         python plot.py \
#             --node $NUM_NODES \
#             --model_name $MODEL_NAME \
#             --setting $SETTING \
#             --workload $WORKLOAD \
#             --task_assignment 'rr' \
#             --retraining_rate $RETRAIN_RATE \
#             --output_dir $OUTPUT_DIR
#     done
# done

MODEL_NAME="DialoGPT-large"
NUM_NODES=4
NUM_GPUs=8
RATE_LAMBDA=-1
SETTING=active
EXPERIMENTS=3
for MEMORY_THRESHOLD in 0.8 0.85 0.9 0.95 0.99; do
# for MEMORY_THRESHOLD in 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99; do
    # for MAX_WAIT in 1 5 9 13 17; do
    for MAX_WAIT in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_alternate/${MODEL_NAME}/${MAX_WAIT}-${MEMORY_THRESHOLD}
        # NaiveMix
        python distributed_dialogpt.py \
            --model_name_or_path "microsoft/$MODEL_NAME" \
            --model_name $MODEL_NAME \
            --num_nodes $NUM_NODES \
            --num_gpus $NUM_GPUs \
            --n_samples $NUM_SAMPLES \
            --rate_lambda $RATE_LAMBDA \
            --workload $WORKLOAD \
            --setting $SETTING \
            --retraining_rate $RETRAIN_RATE \
            --task_assignment 'rr' \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --experiments $EXPERIMENTS \
            --max_wait $MAX_WAIT \
            --memory_threshold $MEMORY_THRESHOLD

        python plot.py \
            --node $NUM_NODES \
            --model_name $MODEL_NAME \
            --setting $SETTING \
            --workload $WORKLOAD \
            --task_assignment 'rr' \
            --retraining_rate $RETRAIN_RATE \
            --output_dir $OUTPUT_DIR
    done
done
