NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
SETTING=active
BASE_DIR=prof_3.1


for NUM_NODES in 4 2; do
    for MODEL_NAME in "DialoGPT-large"; do
        for RATE_LAMBDA in 10 20 30 40 50; do
            OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
            for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                # for TASK_ASSIGNMENT in "workload" "random" "rr" "util"; do
                for TASK_ASSIGNMENT in "util"; do
                    python distributed_dialogpt.py \
                        --model_name_or_path "microsoft/$MODEL_NAME" \
                        --model_name $MODEL_NAME \
                        --num_nodes $NUM_NODES \
                        --n_samples $NUM_SAMPLES \
                        --test_lambda $RATE_LAMBDA \
                        --train_lambda $RATE_LAMBDA \
                        --workload $WORKLOAD \
                        --setting $SETTING \
                        --retraining_rate $RETRAIN_RATE \
                        --task_assignment $TASK_ASSIGNMENT \
                        --output_dir $OUTPUT_DIR \
                        --batch_size $BATCH_SIZE \
                        --memory_threshold $MEMORY_THRESHOLD

                    python plot.py \
                        --node $NUM_NODES \
                        --model_name $MODEL_NAME \
                        --setting $SETTING \
                        --workload $WORKLOAD \
                        --task_assignment $TASK_ASSIGNMENT \
                        --retraining_rate $RETRAIN_RATE \
                        --output_dir $OUTPUT_DIR
                done
            done
        done
    done
done


# NUM_NODES=4
# MODEL_NAME="DialoGPT-large"
# RATE_LAMBDA=10
# RETRAIN_RATE=0.3
# TASK_ASSIGNMENT="workload"
# OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
           
# for ALPHA in 1.0; do
#     python distributed_dialogpt.py \
#         --model_name_or_path "microsoft/$MODEL_NAME" \
#         --model_name $MODEL_NAME \
#         --num_nodes $NUM_NODES \
#         --n_samples $NUM_SAMPLES \
#         --test_lambda $RATE_LAMBDA \
#         --train_lambda $RATE_LAMBDA \
#         --workload $WORKLOAD \
#         --setting $SETTING \
#         --alpha $ALPHA \
#         --retraining_rate $RETRAIN_RATE \
#         --task_assignment $TASK_ASSIGNMENT \
#         --output_dir $OUTPUT_DIR \
#         --batch_size $BATCH_SIZE \
#         --memory_threshold $MEMORY_THRESHOLD

#     python plot.py \
#         --node $NUM_NODES \
#         --model_name $MODEL_NAME \
#         --setting $SETTING \
#         --workload $WORKLOAD \
#         --alpha $ALPHA \
#         --task_assignment $TASK_ASSIGNMENT \
#         --retraining_rate $RETRAIN_RATE \
#         --output_dir $OUTPUT_DIR
# done






