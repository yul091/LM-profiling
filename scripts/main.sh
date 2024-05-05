NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
WORKLOAD=poisson
TASK_ASSIGNMENT=workload # "workload" "random" "rr" "util" "random+" "rr+" "util+"
# LENGTH_DIST=random # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_main
MODEL_NAME="DialoGPT-large" # "DialoGPT-medium" "DialoGPT-small"
ALPHA=0.1
BETA=0.1
EPSILON=0.01
K=0.5
EXPERIMENTS=3 # each experiment run 3 times
NUM_NODES=4


for MODEL_NAME in "DialoGPT-large"; do
    for RATE_LAMBDA in 1 5 10 15 20 25 30; do
    # for RATE_LAMBDA in 1 15 25; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
        # for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
        for RETRAIN_RATE in 0.01 0.05 0.15 0.2 0.25; do
            python distributed_dialogpt.py \
                --model_name_or_path "microsoft/$MODEL_NAME" \
                --model_name $MODEL_NAME \
                --num_nodes $NUM_NODES \
                --n_samples $NUM_SAMPLES \
                --rate_lambda $RATE_LAMBDA \
                --workload $WORKLOAD \
                --setting $SETTING \
                --retraining_rate $RETRAIN_RATE \
                --task_assignment "workload" \
                --output_dir $OUTPUT_DIR \
                --batch_size $BATCH_SIZE \
                --alpha $ALPHA \
                --beta $BETA \
                --epsilon $EPSILON \
                --active_selection "adaptive-0.8" \
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
                --active_selection "adaptive-0.8" \
                --output_dir $OUTPUT_DIR

            # # for TASK_ASSIGNMENT in "workload" "random" "rr" "util"; do
            for TASK_ASSIGNMENT in "workload"; do
                python distributed_dialogpt.py \
                    --model_name_or_path "microsoft/$MODEL_NAME" \
                    --model_name $MODEL_NAME \
                    --num_nodes $NUM_NODES \
                    --n_samples $NUM_SAMPLES \
                    --rate_lambda $RATE_LAMBDA \
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

            done
        done
    done 
done

            # # ISOLATED
            # python distributed_dialogpt.py \
            #     --model_name_or_path "microsoft/$MODEL_NAME" \
            #     --model_name $MODEL_NAME \
            #     --num_nodes $NUM_NODES \
            #     --n_samples $NUM_SAMPLES \
            #     --rate_lambda $RATE_LAMBDA \
            #     --workload $WORKLOAD \
            #     --setting "isolated" \
            #     --retraining_rate $RETRAIN_RATE \
            #     --task_assignment 'rr' \
            #     --output_dir $OUTPUT_DIR \
            #     --batch_size $BATCH_SIZE \
            #     --experiments $EXPERIMENTS \
            #     --memory_threshold $MEMORY_THRESHOLD

            # python plot.py \
            #     --node $NUM_NODES \
            #     --model_name $MODEL_NAME \
            #     --setting "isolated" \
            #     --workload $WORKLOAD \
            #     --task_assignment 'rr' \
            #     --retraining_rate $RETRAIN_RATE \
            #     --output_dir $OUTPUT_DIR

#         done
#     done
# done




RATE_LAMBDA=-1
for MODEL_NAME in "DialoGPT-large"; do
    OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_alternate/$MODEL_NAME
    for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
        python distributed_dialogpt.py \
            --model_name_or_path "microsoft/$MODEL_NAME" \
            --model_name $MODEL_NAME \
            --num_nodes $NUM_NODES \
            --n_samples $NUM_SAMPLES \
            --rate_lambda $RATE_LAMBDA \
            --workload $WORKLOAD \
            --setting $SETTING \
            --retraining_rate $RETRAIN_RATE \
            --task_assignment "workload" \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --alpha $ALPHA \
            --beta $BETA \
            --epsilon $EPSILON \
            --active_selection "adaptive-0.8" \
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
            --active_selection "adaptive-0.8" \
            --output_dir $OUTPUT_DIR

        for TASK_ASSIGNMENT in "workload" "random" "rr" "util"; do
        # for TASK_ASSIGNMENT in "workload"; do
            python distributed_dialogpt.py \
                --model_name_or_path "microsoft/$MODEL_NAME" \
                --model_name $MODEL_NAME \
                --num_nodes $NUM_NODES \
                --n_samples $NUM_SAMPLES \
                --rate_lambda $RATE_LAMBDA \
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
        done

        # ISOLATED
        python distributed_dialogpt.py \
            --model_name_or_path "microsoft/$MODEL_NAME" \
            --model_name $MODEL_NAME \
            --num_nodes $NUM_NODES \
            --n_samples $NUM_SAMPLES \
            --rate_lambda $RATE_LAMBDA \
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


