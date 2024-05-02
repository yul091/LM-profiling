
# NUM_NODES=1
NUM_SAMPLES=1000
BATCH_SIZE=3
# RATE_LAMBDA=10
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
TASK_ASSIGNMENT=rr # "workload" "random" "rr" "util" "random+" "rr+" "util+"
# LENGTH_DIST=random # "ascending" "bursty" "random"
# LENGTH_HETEROGENEITY=200
# RETRAIN_RATE=0.3
BASE_DIR=prof_3.1
MODEL_NAME="DialoGPT-large"

# for NUM_NODES in 1 2 4; do
for NUM_NODES in 1; do
    # for RATE_LAMBDA in 50 30 20 10 5; do
    for RATE_LAMBDA in 5; do
        # for RETRAIN_RATE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        for RETRAIN_RATE in 1.0; do
            if [ $NUM_NODES -eq 1 ]; then
                CUDA_VISIBLE_DEVICES="0,1,2,3" # Use only cuda0-3 when NUM_NODES=1
            else
                CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" # Use all devices otherwise
            fi
            OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME

            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python distributed_dialogpt.py \
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
                --run_mode offline \
                --memory_threshold $MEMORY_THRESHOLD
        done
    done
done