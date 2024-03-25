NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
# "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"

export CUDA_VISIBLE_DEVICES=4,5,6,7 # Use 0, 1, 2, 3 as CUDA device IDs
for NUM_NODES in 1; do
    for MODEL_NAME in "DialoGPT-large"; do
        for RATE_LAMBDA in 10 30; do
            OUTPUT_DIR=prof/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
            for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
                for SETTING in active; do
                    python distributed_dialogpt.py \
                        --model_name_or_path "microsoft/$MODEL_NAME" \
                        --model_name $MODEL_NAME \
                        --num_nodes $NUM_NODES \
                        --n_samples $NUM_SAMPLES \
                        --test_lambda $RATE_LAMBDA \
                        --train_lambda $RATE_LAMBDA \
                        --active_selection 'adaptive' \
                        --workload $WORKLOAD \
                        --setting $SETTING \
                        --retraining_rate $RETRAIN_RATE \
                        --output_dir $OUTPUT_DIR \
                        --batch_size $BATCH_SIZE \
                        --memory_threshold $MEMORY_THRESHOLD

                    python plot.py \
                        --node $NUM_NODES \
                        --model_name $MODEL_NAME \
                        --setting $SETTING \
                        --workload $WORKLOAD \
                        --active_selection 'adaptive' \
                        --retraining_rate $RETRAIN_RATE \
                        --output_dir $OUTPUT_DIR

                    for ACTIVE_SELCTION in 0.2 0.4 0.6 0.8 1.0; do
                        python distributed_dialogpt.py \
                            --model_name_or_path "microsoft/$MODEL_NAME" \
                            --model_name $MODEL_NAME \
                            --num_nodes $NUM_NODES \
                            --n_samples $NUM_SAMPLES \
                            --test_lambda $RATE_LAMBDA \
                            --train_lambda $RATE_LAMBDA \
                            --active_selection $ACTIVE_SELCTION \
                            --workload $WORKLOAD \
                            --setting $SETTING \
                            --retraining_rate $RETRAIN_RATE \
                            --output_dir $OUTPUT_DIR \
                            --batch_size $BATCH_SIZE \
                            --memory_threshold $MEMORY_THRESHOLD

                        python plot.py \
                            --node $NUM_NODES \
                            --model_name $MODEL_NAME \
                            --setting $SETTING \
                            --workload $WORKLOAD \
                            --active_selection $ACTIVE_SELCTION \
                            --retraining_rate $RETRAIN_RATE \
                            --output_dir $OUTPUT_DIR
                    done
                done
            done
        done
    done
done