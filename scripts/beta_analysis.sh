NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
WORKLOAD=poisson
TASK_ASSIGNMENT=workload # "workload" "random" "rr" "util" "random+" "rr+" "util+"
LENGTH_DIST=random # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_new
# MODEL_NAME="DialoGPT-medium" # "DialoGPT-medium" "DialoGPT-small"
ALPHA=0.05
BETA=0.1
EPSILON=0.5
K=0.5
EXPERIMENTS=3 # each experiment run 3 times
NUM_NODES=4


# for RATE_LAMBDA in 35 40 45 50 55 60 65 70 75 80; do
for RATE_LAMBDA in 10 20 30 40 50 60 70 80 90 99; do
    # for RETRAIN_RATE in 0.01 0.05 0.1 0.15 0.2 0.25 0.3; do
    for RETRAIN_RATE in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
        for MODEL_NAME in "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"; do
            OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
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
                --output_dir $OUTPUT_DIR
        done
    done
done


# for RATE_LAMBDA in 90 100; do
#     # for RETRAIN_RATE in 0.01 0.05 0.1 0.15 0.2 0.25 0.3; do
#     for RETRAIN_RATE in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
#         for MODEL_NAME in "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"; do
#             OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
#             python distributed_dialogpt.py \
#                 --model_name_or_path "microsoft/$MODEL_NAME" \
#                 --model_name $MODEL_NAME \
#                 --num_nodes $NUM_NODES \
#                 --n_samples $NUM_SAMPLES \
#                 --rate_lambda $RATE_LAMBDA \
#                 --workload $WORKLOAD \
#                 --setting $SETTING \
#                 --retraining_rate $RETRAIN_RATE \
#                 --task_assignment "workload" \
#                 --output_dir $OUTPUT_DIR \
#                 --batch_size $BATCH_SIZE \
#                 --alpha $ALPHA \
#                 --beta $BETA \
#                 --epsilon $EPSILON \
#                 --k $K \
#                 --experiments $EXPERIMENTS \
#                 --memory_threshold $MEMORY_THRESHOLD

#             python plot.py \
#                 --node $NUM_NODES \
#                 --model_name $MODEL_NAME \
#                 --setting $SETTING \
#                 --workload $WORKLOAD \
#                 --task_assignment "workload" \
#                 --retraining_rate $RETRAIN_RATE \
#                 --alpha $ALPHA \
#                 --beta $BETA \
#                 --epsilon $EPSILON \
#                 --output_dir $OUTPUT_DIR
#         done
#     done
# done

            # for TASK_ASSIGNMENT in 'random' 'rr' 'util' 'workload'; do
            #     python distributed_dialogpt.py \
            #         --model_name_or_path "microsoft/$MODEL_NAME" \
            #         --model_name $MODEL_NAME \
            #         --num_nodes $NUM_NODES \
            #         --n_samples $NUM_SAMPLES \
            #         --rate_lambda $RATE_LAMBDA \
            #         --workload $WORKLOAD \
            #         --setting $SETTING \
            #         --retraining_rate $RETRAIN_RATE \
            #         --task_assignment $TASK_ASSIGNMENT \
            #         --output_dir $OUTPUT_DIR \
            #         --batch_size $BATCH_SIZE \
            #         --alpha $ALPHA \
            #         --beta $BETA \
            #         --epsilon $EPSILON \
            #         --k $K \
            #         --experiments $EXPERIMENTS \
            #         --memory_threshold $MEMORY_THRESHOLD

            #     python plot.py \
            #         --node $NUM_NODES \
            #         --model_name $MODEL_NAME \
            #         --setting $SETTING \
            #         --workload $WORKLOAD \
            #         --task_assignment $TASK_ASSIGNMENT \
            #         --retraining_rate $RETRAIN_RATE \
            #         --alpha $ALPHA \
            #         --beta $BETA \
            #         --epsilon $EPSILON \
            #         --output_dir $OUTPUT_DIR
            # done

            # python distributed_dialogpt.py \
            #     --model_name_or_path "microsoft/$MODEL_NAME" \
            #     --model_name $MODEL_NAME \
            #     --num_nodes $NUM_NODES \
            #     --n_samples $NUM_SAMPLES \
            #     --rate_lambda $RATE_LAMBDA \
            #     --workload $WORKLOAD \
            #     --setting $SETTING \
            #     --retraining_rate $RETRAIN_RATE \
            #     --task_assignment $TASK_ASSIGNMENT \
            #     --output_dir $OUTPUT_DIR \
            #     --batch_size $BATCH_SIZE \
            #     --alpha $ALPHA \
            #     --beta $BETA \
            #     --epsilon $EPSILON \
            #     --k $K \
            #     --experiments $EXPERIMENTS \
            #     --no_prior_profile \
            #     --memory_threshold $MEMORY_THRESHOLD

            # python plot.py \
            #     --node $NUM_NODES \
            #     --model_name $MODEL_NAME \
            #     --setting $SETTING \
            #     --workload $WORKLOAD \
            #     --task_assignment $TASK_ASSIGNMENT \
            #     --retraining_rate $RETRAIN_RATE \
            #     --alpha $ALPHA \
            #     --beta $BETA \
            #     --epsilon $EPSILON \
            #     --no_prior_profile \
            #     --output_dir $OUTPUT_DIR

#         done
#     done
# done