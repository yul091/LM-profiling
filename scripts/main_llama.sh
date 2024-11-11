NUM_SAMPLES=200
BATCH_SIZE=1
MEMORY_THRESHOLD=0.5
SETTING=isolated  # "active" "isolated"
WORKLOAD=poisson
# TASK_ASSIGNMENT=workload  # "workload" "random" "rr" "util" "random+" "rr+" "util+"
# LENGTH_DIST=random  # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_main
ALPHA=0.05
BETA=0.1
EPSILON=0.5
K=0.5
EXPERIMENTS=1  # each experiment run 1 times
NUM_NODES=2
NUM_GPUS=8
TOKEN="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"
# export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

for MODEL_NAME in "Llama-2-7b-chat-hf"; do
    for RATE_LAMBDA in 5; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/${MODEL_NAME}
        for RETRAIN_RATE in 0.5; do
            for TASK_ASSIGNMENT in "rr"; do
                python distributed_llama.py \
                    --model_name_or_path "meta-llama/${MODEL_NAME}" \
                    --access_token $TOKEN \
                    --model_name $MODEL_NAME \
                    --num_gpus $NUM_GPUS \
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
                    --memory_threshold $MEMORY_THRESHOLD \
                    --run_mode 'online'

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

            # # ISOLATED
            # python distributed_llama.py \
            #     --model_name_or_path "meta-llama/$MODEL_NAME" \
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

        done
    done
done


# RATE_LAMBDA=-1
# for MODEL_NAME in "DialoGPT-large" "DialoGPT-medium" "DialoGPT-small"; do
#     OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_alternate/$MODEL_NAME
#     for RETRAIN_RATE in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#         python distributed_llama.py \
#             --model_name_or_path "meta-llama/$MODEL_NAME" \
#             --model_name $MODEL_NAME \
#             --num_nodes $NUM_NODES \
#             --n_samples $NUM_SAMPLES \
#             --rate_lambda $RATE_LAMBDA \
#             --workload $WORKLOAD \
#             --setting $SETTING \
#             --retraining_rate $RETRAIN_RATE \
#             --task_assignment "workload" \
#             --output_dir $OUTPUT_DIR \
#             --batch_size $BATCH_SIZE \
#             --alpha $ALPHA \
#             --beta $BETA \
#             --epsilon $EPSILON \
#             --active_selection "adaptive-0.8" \
#             --k $K \
#             --experiments $EXPERIMENTS \
#             --memory_threshold $MEMORY_THRESHOLD

#         python plot.py \
#             --node $NUM_NODES \
#             --model_name $MODEL_NAME \
#             --setting $SETTING \
#             --workload $WORKLOAD \
#             --task_assignment "workload" \
#             --retraining_rate $RETRAIN_RATE \
#             --alpha $ALPHA \
#             --beta $BETA \
#             --epsilon $EPSILON \
#             --active_selection "adaptive-0.8" \
#             --output_dir $OUTPUT_DIR

#         for TASK_ASSIGNMENT in "workload" "random" "rr" "util"; do
#         # for TASK_ASSIGNMENT in "workload"; do
#             python distributed_llama.py \
#                 --model_name_or_path "meta-llama/$MODEL_NAME" \
#                 --model_name $MODEL_NAME \
#                 --num_nodes $NUM_NODES \
#                 --n_samples $NUM_SAMPLES \
#                 --rate_lambda $RATE_LAMBDA \
#                 --workload $WORKLOAD \
#                 --setting $SETTING \
#                 --retraining_rate $RETRAIN_RATE \
#                 --task_assignment $TASK_ASSIGNMENT \
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
#                 --task_assignment $TASK_ASSIGNMENT \
#                 --retraining_rate $RETRAIN_RATE \
#                 --alpha $ALPHA \
#                 --beta $BETA \
#                 --epsilon $EPSILON \
#                 --output_dir $OUTPUT_DIR
#         done

#         # ISOLATED
#         python distributed_llama.py \
#             --model_name_or_path "meta-llama/$MODEL_NAME" \
#             --model_name $MODEL_NAME \
#             --num_nodes $NUM_NODES \
#             --n_samples $NUM_SAMPLES \
#             --rate_lambda $RATE_LAMBDA \
#             --workload $WORKLOAD \
#             --setting "isolated" \
#             --retraining_rate $RETRAIN_RATE \
#             --task_assignment 'rr' \
#             --output_dir $OUTPUT_DIR \
#             --batch_size $BATCH_SIZE \
#             --experiments $EXPERIMENTS \
#             --memory_threshold $MEMORY_THRESHOLD

#         python plot.py \
#             --node $NUM_NODES \
#             --model_name $MODEL_NAME \
#             --setting "isolated" \
#             --workload $WORKLOAD \
#             --task_assignment 'rr' \
#             --retraining_rate $RETRAIN_RATE \
#             --output_dir $OUTPUT_DIR
#     done
# done


