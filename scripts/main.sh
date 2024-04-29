NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
TASK_ASSIGNMENT=workload # "workload" "random" "rr" "util" "random+" "rr+" "util+"
# LENGTH_DIST=random # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_main
MODEL_NAME="DialoGPT-large"
EPSILON=0.01
ALPHA=1.0
BETA=0.1
K=0.5
EXPERIMENTS=5 # each experiment run 5 times

for NUM_NODES in 4; do
    for RATE_LAMBDA in 100 50 10 2; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
        for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
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
                --task_assignment "workload" \
                --output_dir $OUTPUT_DIR \
                --batch_size $BATCH_SIZE \
                --alpha $ALPHA \
                --beta $BETA \
                --epsilon $EPSILON \
                --active_selection "adaptive-0.9" \
                --k $K \
                --experiments $EXPERIMENTS \
                --memory_threshold $MEMORY_THRESHOLD

            python plot.py \
                --node $NUM_NODES \
                --model_name $MODEL_NAME \
                --setting $SETTING \
                --workload $WORKLOAD \
                --task_assignment "workload" \
                --retraining_rate $RETRAIN_RATE \
                --alpha $ALPHA \
                --beta $BETA \
                --epsilon $EPSILON \
                --active_selection "adaptive-0.9" \
                --output_dir $OUTPUT_DIR


            for TASK_ASSIGNMENT in "workload" "random" "rr" "util"; do
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
                    --alpha $ALPHA \
                    --beta $BETA \
                    --epsilon $EPSILON \
                    --k $K \
                    --experiments $EXPERIMENTS \
                    --memory_threshold $MEMORY_THRESHOLD

                python plot.py \
                    --node $NUM_NODES \
                    --model_name $MODEL_NAME \
                    --setting $SETTING \
                    --workload $WORKLOAD \
                    --task_assignment $TASK_ASSIGNMENT \
                    --retraining_rate $RETRAIN_RATE \
                    --alpha $ALPHA \
                    --beta $BETA \
                    --epsilon $EPSILON \
                    --output_dir $OUTPUT_DIR

            # ISOLATED
            python distributed_dialogpt.py \
                --model_name_or_path "microsoft/$MODEL_NAME" \
                --model_name $MODEL_NAME \
                --num_nodes $NUM_NODES \
                --n_samples $NUM_SAMPLES \
                --test_lambda $RATE_LAMBDA \
                --train_lambda $RATE_LAMBDA \
                --workload $WORKLOAD \
                --setting "isolated" \
                --retraining_rate $RETRAIN_RATE \
                --task_assignment 'rr' \
                --output_dir $OUTPUT_DIR \
                --batch_size $BATCH_SIZE \
                --experiments $EXPERIMENTS \
                --memory_threshold $MEMORY_THRESHOLD

            python plot.py \
                --node $NUM_NODES \
                --model_name $MODEL_NAME \
                --setting "isolated" \
                --workload $WORKLOAD \
                --task_assignment 'rr' \
                --retraining_rate $RETRAIN_RATE \
                --output_dir $OUTPUT_DIR
            done
        done
    done
done






