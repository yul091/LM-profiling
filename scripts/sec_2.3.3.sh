NUM_SAMPLES=1000
BATCH_SIZE=3
WORKLOAD=poisson
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
TASK_ASSIGNMENT=rr # "workload" "random" "rr" "util" "random+" "rr+" "util+"
BASE_DIR=prof_2.3.3
MODEL_NAME="DialoGPT-large"
RETRAIN_RATE=0.5
EXPERIMENTS=5 # each experiment run 5 times
export CUDA_VISIBLE_DEVICES=4,5,6,7

for NUM_NODES in 1; do
    for RATE_LAMBDA in 50 10 1; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME

        # python distributed_dialogpt.py \
        #     --model_name_or_path "microsoft/$MODEL_NAME" \
        #     --model_name $MODEL_NAME \
        #     --num_nodes $NUM_NODES \
        #     --n_samples $NUM_SAMPLES \
        #     --test_lambda $RATE_LAMBDA \
        #     --train_lambda $RATE_LAMBDA \
        #     --workload $WORKLOAD \
        #     --setting $SETTING \
        #     --retraining_rate $RETRAIN_RATE \
        #     --task_assignment $TASK_ASSIGNMENT \
        #     --output_dir $OUTPUT_DIR \
        #     --batch_size $BATCH_SIZE \
        #     --experiments $EXPERIMENTS \
        #     --memory_threshold $MEMORY_THRESHOLD

        # python plot.py \
        #     --node $NUM_NODES \
        #     --model_name $MODEL_NAME \
        #     --setting $SETTING \
        #     --workload $WORKLOAD \
        #     --task_assignment $TASK_ASSIGNMENT \
        #     --retraining_rate $RETRAIN_RATE \
        #     --output_dir $OUTPUT_DIR

        for ACTIVE_SELECTION in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
            # for POLICY in 'random' 'length' 'loss' 'first'; do
            for POLICY in 'first'; do
                SELECTION_POLICY="${POLICY}-${ACTIVE_SELECTION}"
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
                    --active_selection $SELECTION_POLICY \
                    --experiments $EXPERIMENTS \
                    --memory_threshold $MEMORY_THRESHOLD

                python plot.py \
                    --node $NUM_NODES \
                    --model_name $MODEL_NAME \
                    --setting $SETTING \
                    --workload $WORKLOAD \
                    --task_assignment $TASK_ASSIGNMENT \
                    --retraining_rate $RETRAIN_RATE \
                    --active_selection $SELECTION_POLICY \
                    --output_dir $OUTPUT_DIR
            done
        done
    done
done

