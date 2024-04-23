
NUM_NODES=4
NUM_SAMPLES=1000
BATCH_SIZE=3
# RATE_LAMBDA=10
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
TASK_ASSIGNMENT=random # "workload" "random" "rr" "util" "random+" "rr+" "util+"
LENGTH_DIST=random # "ascending" "bursty" "random"
LENGTH_HETEROGENEITY=200
RETRAIN_RATE=0.3
BASE_DIR=prof_3.1
MODEL_NAME="DialoGPT-large"
OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME

for RATE_LAMBDA in 30 25 20 15 10 5 1; do
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
        --length_heterogeneity $LENGTH_HETEROGENEITY \
        --length_distribution $LENGTH_DIST \
        --memory_threshold $MEMORY_THRESHOLD
done