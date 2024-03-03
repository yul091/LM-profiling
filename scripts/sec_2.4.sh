NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
# LOAD_BALANCING=workload
# LENGTH_DISTRIBUTION=ascending # ascending, descending, random, bursty
MODEL_NAME="DialoGPT-large" # "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"

for NUM_NODES in 2 4; do
    for RATE_LAMBDA in 10 15 20 25 30 50; do
        OUTPUT_DIR=prof/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
        for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            for SETTING in "active" "isolated"; do
                for LENGTH_DISTRIBUTION in "ascending" "descending" "random" "bursty"; do
                    python distributed_dialogpt.py \
                        --model_name_or_path "microsoft/$MODEL_NAME" \
                        --model_name $MODEL_NAME \
                        --num_nodes $NUM_NODES \
                        --n_samples $NUM_SAMPLES \
                        --rate_lambda $RATE_LAMBDA \
                        --workload $WORKLOAD \
                        --setting $SETTING \
                        --retraining_rate $RETRAIN_RATE \
                        --output_dir $OUTPUT_DIR \
                        --batch_size $BATCH_SIZE \
                        --length_distribution $LENGTH_DISTRIBUTION \
                        --memory_threshold $MEMORY_THRESHOLD

                    python plot.py \
                        --node $NUM_NODES \
                        --model_name $MODEL_NAME \
                        --setting $SETTING \
                        --workload $WORKLOAD \
                        --retraining_rate $RETRAIN_RATE \
                        --length_distribution $LENGTH_DISTRIBUTION \
                        --output_dir $OUTPUT_DIR
                done
            done
        done
    done
done
