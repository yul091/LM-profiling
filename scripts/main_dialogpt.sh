NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
WORKLOAD=poisson
# TASK_ASSIGNMENT=workload # "workload" "rr" "util" "random+" "rr+" "util+"
# LENGTH_DIST=random # "ascending" "bursty" "descending"
BASE_DIR=prof_new_new
# MODEL_NAME="DialoGPT-large" # "DialoGPT-medium" "DialoGPT-small"
ALPHA=0.15
BETA=0.05
EPSILON=0.2
K=0.5
EXPERIMENTS=3 # each experiment run 3 times
# NUM_NODES=4
PROFILING_DIR="nvidia-profiling_new"
mkdir -p ${PROFILING_DIR}

# "DialoGPT-small" "DialoGPT-medium" 
for NUM_NODES in 4; do
    for MODEL_NAME in  "DialoGPT-large" "DialoGPT-medium" "DialoGPT-small"; do
        for RATE_LAMBDA in 4 8 12 16 20 24 28 32 -1; do
            OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/${MODEL_NAME}
            for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
                for TASK_ASSIGNMENT in "rr" "util" "workload"; do
                    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/${TASK_ASSIGNMENT}_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate.csv &
                    NVIDIA_SMI_PID=$!
                    python distributed_dialogpt.py \
                        --model_name_or_path "microsoft/${MODEL_NAME}" \
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
                    
                    kill $NVIDIA_SMI_PID
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
                nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/separate_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate.csv &
                NVIDIA_SMI_PID=$!
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

                kill $NVIDIA_SMI_PID
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


# NUM_SAMPLES=3000
# NUM_NODES=4
# # Allocated nodes experiments
# # for RATE_LAMBDA in 10 20 30 40 50 60 70 80 90; do
#     # for RETRAIN_RATE in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
# for RATE_LAMBDA in 50; do
#     for RETRAIN_RATE in 0.3; do
#         # for MODEL_NAME in "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"; do
#         for MODEL_NAME in "DialoGPT-large"; do
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

