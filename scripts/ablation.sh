NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
WORKLOAD=poisson
TASK_ASSIGNMENT=workload # "workload" "random" "rr" "util" "random+" "rr+" "util+"
LENGTH_DIST=random # "ascending" "bursty" "random" "descending"
BASE_DIR=res_ablation
MODEL_NAME="DialoGPT-large" # "DialoGPT-medium" "DialoGPT-small"
ALPHA=0.05
BETA=0.1
EPSILON=0.5
K=0.5
EXPERIMENTS=3 # each experiment run 3 times
NUM_NODES=4
RATE_LAMBDA=-1
OFFLINE_DIR=profile_new
PROFILING_DIR="nvidia-profiling"
mkdir -p ${PROFILING_DIR}

# for RATE_LAMBDA in 30 25 20 15 10 5 1; do
for MODEL_NAME in "DialoGPT-large"; do
    # for RETRAIN_RATE in 0.01 0.05 0.1 0.15 0.2 0.25 0.3; do
    for RETRAIN_RATE in 0.1 0.3 0.7 0.9; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_alternate/$MODEL_NAME
        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/${TASK_ASSIGNMENT}_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate_no-prior-profile.csv &
        NVIDIA_SMI_PID=$!
        python distributed_dialogpt.py \
            --model_name_or_path "microsoft/$MODEL_NAME" \
            --model_name $MODEL_NAME \
            --num_nodes $NUM_NODES \
            --n_samples $NUM_SAMPLES \
            --rate_lambda $RATE_LAMBDA \
            --workload $WORKLOAD \
            --setting $SETTING \
            --retraining_rate $RETRAIN_RATE \
            --task_assignment 'workload' \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --alpha $ALPHA \
            --beta $BETA \
            --epsilon $EPSILON \
            --experiments $EXPERIMENTS \
            --profile_dir $OFFLINE_DIR \
            --no_prior_profile \
            --memory_threshold $MEMORY_THRESHOLD
        kill $NVIDIA_SMI_PID

        python plot.py \
            --node $NUM_NODES \
            --model_name $MODEL_NAME \
            --setting $SETTING \
            --workload $WORKLOAD \
            --task_assignment 'workload' \
            --retraining_rate $RETRAIN_RATE \
            --alpha $ALPHA \
            --beta $BETA \
            --epsilon $EPSILON \
            --no_prior_profile \
            --output_dir $OUTPUT_DIR

        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/${TASK_ASSIGNMENT}_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate_no-memory-check.csv &
        NVIDIA_SMI_PID=$!
        python distributed_dialogpt.py \
            --model_name_or_path "microsoft/$MODEL_NAME" \
            --model_name $MODEL_NAME \
            --num_nodes $NUM_NODES \
            --n_samples $NUM_SAMPLES \
            --rate_lambda $RATE_LAMBDA \
            --workload $WORKLOAD \
            --setting $SETTING \
            --retraining_rate $RETRAIN_RATE \
            --task_assignment 'workload' \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --alpha $ALPHA \
            --beta $BETA \
            --epsilon $EPSILON \
            --experiments $EXPERIMENTS \
            --profile_dir $OFFLINE_DIR \
            --no_memory_check \
            --memory_threshold $MEMORY_THRESHOLD
        kill $NVIDIA_SMI_PID

        python plot.py \
            --node $NUM_NODES \
            --model_name $MODEL_NAME \
            --setting $SETTING \
            --workload $WORKLOAD \
            --task_assignment 'workload' \
            --retraining_rate $RETRAIN_RATE \
            --alpha $ALPHA \
            --beta $BETA \
            --epsilon $EPSILON \
            --no_memory_check \
            --output_dir $OUTPUT_DIR

        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/${TASK_ASSIGNMENT}_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate.csv &
        NVIDIA_SMI_PID=$!
        python distributed_dialogpt.py \
            --model_name_or_path "microsoft/$MODEL_NAME" \
            --model_name $MODEL_NAME \
            --num_nodes $NUM_NODES \
            --n_samples $NUM_SAMPLES \
            --rate_lambda $RATE_LAMBDA \
            --workload $WORKLOAD \
            --setting $SETTING \
            --retraining_rate $RETRAIN_RATE \
            --task_assignment 'workload' \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --alpha $ALPHA \
            --beta $BETA \
            --epsilon $EPSILON \
            --experiments $EXPERIMENTS \
            --profile_dir $OFFLINE_DIR \
            --memory_threshold $MEMORY_THRESHOLD
        kill $NVIDIA_SMI_PID

        python plot.py \
            --node $NUM_NODES \
            --model_name $MODEL_NAME \
            --setting $SETTING \
            --workload $WORKLOAD \
            --task_assignment 'workload' \
            --retraining_rate $RETRAIN_RATE \
            --alpha $ALPHA \
            --beta $BETA \
            --epsilon $EPSILON \
            --output_dir $OUTPUT_DIR
        done
    done
done


# RATE_LAMBDA=-1
# for MODEL_NAME in "DialoGPT-large" "DialoGPT-medium" "DialoGPT-small"; do
#     for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
#         OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_alternate/$MODEL_NAME
#         python distributed_dialogpt.py \
#             --model_name_or_path "microsoft/$MODEL_NAME" \
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
#             --no_prior_profile \
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
#             --no_prior_profile \
#             --output_dir $OUTPUT_DIR

#         for TASK_ASSIGNMENT in 'workload'; do
#             python distributed_dialogpt.py \
#                 --model_name_or_path "microsoft/$MODEL_NAME" \
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
#                 --no_prior_profile \
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
#                 --no_prior_profile \
#                 --output_dir $OUTPUT_DIR
#         done
#     done
# done