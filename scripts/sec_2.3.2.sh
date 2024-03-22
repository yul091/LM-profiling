NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
# LOAD_BALANCING=workload

for NUM_NODES in 2 4; do
    for MODEL_NAME in "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"; do
        for RATE_LAMBDA in 10 20 30; do
            OUTPUT_DIR=prof/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
            for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                for SETTING in active isolated; do
                    # for LENGTH_VAR in 0 200; do
                    python distributed_dialogpt.py \
                        --model_name_or_path "microsoft/$MODEL_NAME" \
                        --model_name $MODEL_NAME \
                        --num_nodes $NUM_NODES \
                        --n_samples $NUM_SAMPLES \
                        --test_lambda $RATE_LAMBDA \
                        --train_lambda $RATE_LAMBDA \
                        --length_heterogeneity 0 \
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
                        --length_heterogeneity 0 \
                        --retraining_rate $RETRAIN_RATE \
                        --output_dir $OUTPUT_DIR

                    for LENGTH_DIST in "ascending" "descending" "bursty" "random"; do
                        python distributed_dialogpt.py \
                            --model_name_or_path "microsoft/$MODEL_NAME" \
                            --model_name $MODEL_NAME \
                            --num_nodes $NUM_NODES \
                            --n_samples $NUM_SAMPLES \
                            --test_lambda $RATE_LAMBDA \
                            --train_lambda $RATE_LAMBDA \
                            --length_heterogeneity 200 \
                            --length_distribution $LENGTH_DIST \
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
                            --length_heterogeneity 200 \
                            --length_distribution $LENGTH_DIST \
                            --retraining_rate $RETRAIN_RATE \
                            --output_dir $OUTPUT_DIR
                    done
                done
            done
        done
    done
done

