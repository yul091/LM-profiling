NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
TASK_ASSIGNMENT=workload # "workload" "random" "rr" "util"
LENGTH_DIST=random # "ascending" "bursty" "random"
LENGTH_HETEROGENEITY=200
BASE_DIR=prof_3.1
MODEL_NAME="DialoGPT-large"
EPSILON=0.05
ALPHA=1.0
BETA=0.1

for NUM_NODES in 4; do
    # for MODEL_NAME in "DialoGPT-large"; do
    # for LENGTH_DIST in "random"; do
    for RATE_LAMBDA in 30 25 20 15 10 5 1; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
        for RETRAIN_RATE in 0.01 0.05 0.1 0.15 0.2 0.25 0.3; do
            # for LENGTH_HETEROGENEITY in 0 10 50 100 200; do
            # for TASK_ASSIGNMENT in "random" "rr" "util" "workload"; do
            for EPSILON in  0.001 0.005 0.01 0.05 0.1 0.5; do
                for ACTIVE_SELECTION in 'adaptive' 'random' 'first' 'last'; do
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
                        --active_selection $ACTIVE_SELECTION \
                        --length_heterogeneity $LENGTH_HETEROGENEITY \
                        --length_distribution $LENGTH_DIST \
                        --memory_threshold $MEMORY_THRESHOLD

                    python plot.py \
                        --node $NUM_NODES \
                        --model_name $MODEL_NAME \
                        --setting $SETTING \
                        --workload $WORKLOAD \
                        --task_assignment $TASK_ASSIGNMENT \
                        --retraining_rate $RETRAIN_RATE \
                        --length_heterogeneity $LENGTH_HETEROGENEITY \
                        --length_distribution $LENGTH_DIST \
                        --active_selection $ACTIVE_SELECTION \
                        --alpha $ALPHA \
                        --beta $BETA \
                        --epsilon $EPSILON \
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






