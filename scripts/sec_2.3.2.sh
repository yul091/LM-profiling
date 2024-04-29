NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
TASK_ASSIGNMENT=rr # "workload" "random" "rr" "util" "random+" "rr+" "util+"
LENGTH_DIST=ascending # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_2.3.2
MODEL_NAME="DialoGPT-large"
EXPERIMENTS=5 # each experiment run 5 times
export CUDA_VISIBLE_DEVICES=0,1,2,3

for NUM_NODES in 1; do
    for RATE_LAMBDA in 50 10 1; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
        for RETRAIN_RATE in 0.3 0.5 0.7; do
            for LENGTH_HETEROGENEITY in 0 10 20 30 40 50 80 100 120 150 200; do
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
                    --experiments $EXPERIMENTS \
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
                    --output_dir $OUTPUT_DIR
            done
        done
    done
done






