NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
TASK_ASSIGNMENT=rr # "workload" "random" "rr" "util" "random+" "rr+" "util+"
BASE_DIR=prof_2.3.1
MODEL_NAME="DialoGPT-large"

for NUM_NODES in 4; do
    for RATE_LAMBDA in 50 40 30 25 20 15 10 5 1; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
        for RETRAIN_RATE in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
            for SETTING in "active" "isolated"; do
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






