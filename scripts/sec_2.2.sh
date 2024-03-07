NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
# MODEL_NAME="DialoGPT-large"
# SETTING=isolated

for NUM_NODES in 4 2; do
    for MODEL_NAME in "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"; do
        for RATE_LAMBDA in 10 20 30 40 50; do
            OUTPUT_DIR=prof/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
            for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                for SETTING in isolated active; do
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
                        --memory_threshold $MEMORY_THRESHOLD

                    python plot.py \
                        --node $NUM_NODES \
                        --model_name $MODEL_NAME \
                        --setting $SETTING \
                        --workload $WORKLOAD \
                        --retraining_rate $RETRAIN_RATE \
                        --output_dir $OUTPUT_DIR
                done
            done
        done
    done
done


# SETTING=isolated
# for NUM_NODES in 4; do
#     for MODEL_NAME in "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"; do
#         for RATE_LAMBDA in 5 10 15 20 25 30 50; do
#             OUTPUT_DIR=prof/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
#             for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#                 for ISOLATED_SPLIT in 0.5 -1; do
#                     python distributed_dialogpt.py \
#                         --model_name_or_path "microsoft/$MODEL_NAME" \
#                         --model_name $MODEL_NAME \
#                         --num_nodes $NUM_NODES \
#                         --n_samples $NUM_SAMPLES \
#                         --rate_lambda $RATE_LAMBDA \
#                         --workload $WORKLOAD \
#                         --isolated_split $ISOLATED_SPLIT \
#                         --setting $SETTING \
#                         --retraining_rate $RETRAIN_RATE \
#                         --output_dir $OUTPUT_DIR \
#                         --batch_size $BATCH_SIZE \
#                         --memory_threshold $MEMORY_THRESHOLD

#                     python plot.py \
#                         --node $NUM_NODES \
#                         --model_name $MODEL_NAME \
#                         --setting $SETTING \
#                         --workload $WORKLOAD \
#                         --isolated_split $ISOLATED_SPLIT \
#                         --retraining_rate $RETRAIN_RATE \
#                         --output_dir $OUTPUT_DIR
#                 done
#             done
#         done
#     done
# done

# for NUM_NODES in 4; do
#     OUTPUT_DIR=prof/${NUM_NODES}_node/varying_lambda/$MODEL_NAME
#     for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#         python distributed_dialogpt.py \
#             --model_name_or_path "microsoft/$MODEL_NAME" \
#             --model_name $MODEL_NAME \
#             --num_nodes $NUM_NODES \
#             --n_samples $NUM_SAMPLES \
#             --workload $WORKLOAD \
#             --setting active \
#             --retraining_rate $RETRAIN_RATE \
#             --output_dir $OUTPUT_DIR \
#             --batch_size $BATCH_SIZE \
#             --memory_threshold $MEMORY_THRESHOLD

#         python plot.py \
#             --node $NUM_NODES \
#             --model_name $MODEL_NAME \
#             --setting active \
#             --workload $WORKLOAD \
#             --retraining_rate $RETRAIN_RATE \
#             --output_dir $OUTPUT_DIR

#         for ISOLATED_SPLIT in 0 0.5 -1.0; do
#             python distributed_dialogpt.py \
#                 --model_name_or_path "microsoft/$MODEL_NAME" \
#                 --model_name $MODEL_NAME \
#                 --num_nodes $NUM_NODES \
#                 --n_samples $NUM_SAMPLES \
#                 --workload $WORKLOAD \
#                 --isolated_split $ISOLATED_SPLIT \
#                 --setting isolated \
#                 --retraining_rate $RETRAIN_RATE \
#                 --output_dir $OUTPUT_DIR \
#                 --batch_size $BATCH_SIZE \
#                 --memory_threshold $MEMORY_THRESHOLD

#             python plot.py \
#                 --node $NUM_NODES \
#                 --model_name $MODEL_NAME \
#                 --setting isolated \
#                 --workload $WORKLOAD \
#                 --isolated_split $ISOLATED_SPLIT \
#                 --retraining_rate $RETRAIN_RATE \
#                 --output_dir $OUTPUT_DIR
#             done
#         done
#     done
# done
