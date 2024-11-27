NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
WORKLOAD=poisson
# TASK_ASSIGNMENT=util # "workload" "random" "rr" "util" "random+" "rr+" "util+"
LENGTH_DIST=bursty # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_lh
ALPHA=0.05
BETA=0.1
EPSILON=0.5
K=0.5
EXPERIMENTS=1 # each experiment run 3 times
NUM_NODES=4
PROFILING_DIR="nvidia-profiling"
mkdir -p ${PROFILING_DIR}
RATE_LAMBDA=50


# for MODEL_NAME in "DialoGPT-small" "DialoGPT-medium" "DialoGPT-large"; do
for MODEL_NAME in "DialoGPT-large"; do
    OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
    for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
        # for LENGTH_HETEROGENEITY in  150 100 50 10; do
        for LENGTH_HETEROGENEITY in 200 175 150 125 100 75 50 25 10; do
            for TASK_ASSIGNMENT in "rr" "util"; do
                nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/${TASK_ASSIGNMENT}_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate_LH-${LENGTH_HETEROGENEITY}.csv &
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
                    --task_assignment $TASK_ASSIGNMENT \
                    --output_dir $OUTPUT_DIR \
                    --batch_size $BATCH_SIZE \
                    --alpha $ALPHA \
                    --beta $BETA \
                    --epsilon $EPSILON \
                    --k $K \
                    --length_heterogeneity $LENGTH_HETEROGENEITY \
                    --length_distribution $LENGTH_DIST \
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
                    --length_heterogeneity $LENGTH_HETEROGENEITY \
                    --length_distribution $LENGTH_DIST \
                    --output_dir $OUTPUT_DIR
            done

            # ISOLATED
            nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/separate_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate_LH-${LENGTH_HETEROGENEITY}.csv &
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
                --length_heterogeneity $LENGTH_HETEROGENEITY \
                --length_distribution $LENGTH_DIST \
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
                --length_heterogeneity $LENGTH_HETEROGENEITY \
                --length_distribution $LENGTH_DIST \
                --output_dir $OUTPUT_DIR

        done
    done
done


# RATE_LAMBDA=-1
# for NUM_NODES in 4; do
#     OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_alternate/$MODEL_NAME
#     for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#         # for ALPHA in 0.5 0.1 0.01; do
#             # for BETA in 0.1 0.01 0.001; do
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
#             --length_heterogeneity $LENGTH_HETEROGENEITY \
#             --length_distribution $LENGTH_DIST \
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
#             --length_heterogeneity $LENGTH_HETEROGENEITY \
#             --length_distribution $LENGTH_DIST \
#             --active_selection "adaptive-0.8" \
#             --output_dir $OUTPUT_DIR

#         for TASK_ASSIGNMENT in "workload" "random" "rr" "util"; do
#         # for TASK_ASSIGNMENT in "workload"; do
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
#                 --length_heterogeneity $LENGTH_HETEROGENEITY \
#                 --length_distribution $LENGTH_DIST \
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
#                 --length_heterogeneity $LENGTH_HETEROGENEITY \
#                 --length_distribution $LENGTH_DIST \
#                 --output_dir $OUTPUT_DIR
#         done

#         # ISOLATED
#         python distributed_dialogpt.py \
#             --model_name_or_path "microsoft/$MODEL_NAME" \
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
#             --length_heterogeneity $LENGTH_HETEROGENEITY \
#             --length_distribution $LENGTH_DIST \
#             --experiments $EXPERIMENTS \
#             --memory_threshold $MEMORY_THRESHOLD

#         python plot.py \
#             --node $NUM_NODES \
#             --model_name $MODEL_NAME \
#             --setting "isolated" \
#             --workload $WORKLOAD \
#             --task_assignment 'rr' \
#             --retraining_rate $RETRAIN_RATE \
#             --length_heterogeneity $LENGTH_HETEROGENEITY \
#             --length_distribution $LENGTH_DIST \
#             --output_dir $OUTPUT_DIR
#     done
# done



# for NUM_NODES in 4; do
#     for RATE_LAMBDA in 5 10 30 50; do
#         OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
#         for RETRAIN_RATE in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
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
#                 --active_selection "adaptive-0.8" \
#                 --k $K \
#                 --length_heterogeneity $LENGTH_HETEROGENEITY \
#                 --length_distribution $LENGTH_DIST \
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
#                 --active_selection "adaptive-0.8" \
#                 --length_heterogeneity $LENGTH_HETEROGENEITY \
#                 --length_distribution $LENGTH_DIST \
#                 --output_dir $OUTPUT_DIR

#             for TASK_ASSIGNMENT in "workload" "random" "rr" "util"; do
#             # for TASK_ASSIGNMENT in "workload"; do
#                 python distributed_dialogpt.py \
#                     --model_name_or_path "microsoft/$MODEL_NAME" \
#                     --model_name $MODEL_NAME \
#                     --num_nodes $NUM_NODES \
#                     --n_samples $NUM_SAMPLES \
#                     --rate_lambda $RATE_LAMBDA \
#                     --workload $WORKLOAD \
#                     --setting $SETTING \
#                     --retraining_rate $RETRAIN_RATE \
#                     --task_assignment $TASK_ASSIGNMENT \
#                     --output_dir $OUTPUT_DIR \
#                     --batch_size $BATCH_SIZE \
#                     --alpha $ALPHA \
#                     --beta $BETA \
#                     --epsilon $EPSILON \
#                     --k $K \
#                     --length_heterogeneity $LENGTH_HETEROGENEITY \
#                     --length_distribution $LENGTH_DIST \
#                     --experiments $EXPERIMENTS \
#                     --memory_threshold $MEMORY_THRESHOLD

#                 python plot.py \
#                     --node $NUM_NODES \
#                     --model_name $MODEL_NAME \
#                     --setting $SETTING \
#                     --workload $WORKLOAD \
#                     --task_assignment $TASK_ASSIGNMENT \
#                     --retraining_rate $RETRAIN_RATE \
#                     --alpha $ALPHA \
#                     --beta $BETA \
#                     --epsilon $EPSILON \
#                     --length_heterogeneity $LENGTH_HETEROGENEITY \
#                     --length_distribution $LENGTH_DIST \
#                     --output_dir $OUTPUT_DIR
#             done

#             # ISOLATED
#             python distributed_dialogpt.py \
#                 --model_name_or_path "microsoft/$MODEL_NAME" \
#                 --model_name $MODEL_NAME \
#                 --num_nodes $NUM_NODES \
#                 --n_samples $NUM_SAMPLES \
#                 --rate_lambda $RATE_LAMBDA \
#                 --workload $WORKLOAD \
#                 --setting "isolated" \
#                 --retraining_rate $RETRAIN_RATE \
#                 --task_assignment 'rr' \
#                 --output_dir $OUTPUT_DIR \
#                 --batch_size $BATCH_SIZE \
#                 --length_heterogeneity $LENGTH_HETEROGENEITY \
#                 --length_distribution $LENGTH_DIST \
#                 --experiments $EXPERIMENTS \
#                 --memory_threshold $MEMORY_THRESHOLD

#             python plot.py \
#                 --node $NUM_NODES \
#                 --model_name $MODEL_NAME \
#                 --setting "isolated" \
#                 --workload $WORKLOAD \
#                 --task_assignment 'rr' \
#                 --retraining_rate $RETRAIN_RATE \
#                 --length_heterogeneity $LENGTH_HETEROGENEITY \
#                 --length_distribution $LENGTH_DIST \
#                 --output_dir $OUTPUT_DIR

#         done
#     done
# done






