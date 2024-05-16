NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.5
SETTING=active # "active" "isolated"
WORKLOAD=poisson
TASK_ASSIGNMENT=workload # "workload" "random" "rr" "util" "random+" "rr+" "util+"
LENGTH_DIST=bursty # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_lh
# MODEL_NAME="DialoGPT-large" # "DialoGPT-medium" "DialoGPT-small"
# ALPHA=0.05
ALPHA=0.1
BETA=0.1
# EPSILON=0.5
EPSILON=0.01
K=0.5
EXPERIMENTS=3 # each experiment run 3 times
NUM_NODES=4
RATE_LAMBDA=50 # alternate lambda
RETRAIN_RATE=0.5

for MODEL_NAME in "DialoGPT-large" "DialoGPT-medium" "DialoGPT-small"; do
    # for LENGTH_DIST in "ascending" "bursty" "random"; do
    for RETRAIN_RATE in 0.3; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
        # OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_alternate/$MODEL_NAME
        # for ACTIVE_SELECT in "adaptive-0.0" "adaptive-0.1" "adaptive-0.2" "adaptive-0.3" "adaptive-0.4" "adaptive-0.5" "adaptive-0.6" "adaptive-0.7" "adaptive-0.8" "adaptive-0.9" "adaptive-1.0"; do
        #     python distributed_dialogpt.py \
        #         --model_name_or_path "microsoft/$MODEL_NAME" \
        #         --model_name $MODEL_NAME \
        #         --num_nodes $NUM_NODES \
        #         --n_samples $NUM_SAMPLES \
        #         --rate_lambda $RATE_LAMBDA \
        #         --workload $WORKLOAD \
        #         --setting $SETTING \
        #         --retraining_rate $RETRAIN_RATE \
        #         --task_assignment "workload" \
        #         --output_dir $OUTPUT_DIR \
        #         --batch_size $BATCH_SIZE \
        #         --alpha $ALPHA \
        #         --beta $BETA \
        #         --epsilon $EPSILON \
        #         --active_selection $ACTIVE_SELECT \
        #         --k $K \
        #         --experiments $EXPERIMENTS \
        #         --memory_threshold $MEMORY_THRESHOLD

        #     python plot.py \
        #         --node $NUM_NODES \
        #         --model_name $MODEL_NAME \
        #         --setting $SETTING \
        #         --workload $WORKLOAD \
        #         --task_assignment "workload" \
        #         --retraining_rate $RETRAIN_RATE \
        #         --alpha $ALPHA \
        #         --beta $BETA \
        #         --epsilon $EPSILON \
        #         --active_selection $ACTIVE_SELECT \
        #         --output_dir $OUTPUT_DIR

        # done
        for LH in 10 30 50 100 150 200; do
            for TASK_ASSIGNMENT in 'rr' 'workload'; do 
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
                    --length_heterogeneity $LH \
                    --length_distribution $LENGTH_DIST \
                    --batch_size $BATCH_SIZE \
                    --alpha $ALPHA \
                    --beta $BETA \
                    --epsilon $EPSILON \
                    --experiments $EXPERIMENTS \
                    --memory_threshold $MEMORY_THRESHOLD

                python plot.py \
                    --node $NUM_NODES \
                    --model_name $MODEL_NAME \
                    --setting $SETTING \
                    --workload $WORKLOAD \
                    --alpha $ALPHA \
                    --beta $BETA \
                    --epsilon $EPSILON \
                    --task_assignment $TASK_ASSIGNMENT \
                    --length_heterogeneity $LH \
                    --length_distribution $LENGTH_DIST \
                    --retraining_rate $RETRAIN_RATE \
                    --output_dir $OUTPUT_DIR
            done

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
                --active_selection "adaptive-0.8" \
                --length_heterogeneity $LH \
                --length_distribution $LENGTH_DIST \
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
                --length_heterogeneity $LH \
                --length_distribution $LENGTH_DIST \
                --active_selection "adaptive-0.8" \
                --output_dir $OUTPUT_DIR

            # ISOLATED
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
                --length_heterogeneity $LH \
                --length_distribution $LENGTH_DIST \
                --experiments $EXPERIMENTS \
                --memory_threshold $MEMORY_THRESHOLD

            python plot.py \
                --node $NUM_NODES \
                --model_name $MODEL_NAME \
                --setting "isolated" \
                --workload $WORKLOAD \
                --task_assignment 'rr' \
                --length_heterogeneity $LH \
                --length_distribution $LENGTH_DIST \
                --retraining_rate $RETRAIN_RATE \
                --output_dir $OUTPUT_DIR

        done

        # for EPSILON in 5.0 1.0 0.5 0.1 0.05; do
        # for EPSILON in 10.0; do
        #     python distributed_dialogpt.py \
        #         --model_name_or_path "microsoft/$MODEL_NAME" \
        #         --model_name $MODEL_NAME \
        #         --num_nodes $NUM_NODES \
        #         --n_samples $NUM_SAMPLES \
        #         --rate_lambda $RATE_LAMBDA \
        #         --workload $WORKLOAD \
        #         --setting $SETTING \
        #         --retraining_rate $RETRAIN_RATE \
        #         --task_assignment "workload" \
        #         --output_dir $OUTPUT_DIR \
        #         --batch_size $BATCH_SIZE \
        #         --alpha 0.1 \
        #         --beta 0.1 \
        #         --epsilon $EPSILON \
        #         --active_selection "adaptive-0.8" \
        #         --k $K \
        #         --experiments $EXPERIMENTS \
        #         --memory_threshold $MEMORY_THRESHOLD

        #     python plot.py \
        #         --node $NUM_NODES \
        #         --model_name $MODEL_NAME \
        #         --setting $SETTING \
        #         --workload $WORKLOAD \
        #         --task_assignment "workload" \
        #         --retraining_rate $RETRAIN_RATE \
        #         --alpha 0.1 \
        #         --beta 0.1 \
        #         --epsilon $EPSILON \
        #         --active_selection "adaptive-0.8" \
        #         --output_dir $OUTPUT_DIR
        # done

        # for ALPHA in 5.0 1.0 0.5 0.05 0.01; do
        # for ALPHA in 10.0; do
        #     python distributed_dialogpt.py \
        #         --model_name_or_path "microsoft/$MODEL_NAME" \
        #         --model_name $MODEL_NAME \
        #         --num_nodes $NUM_NODES \
        #         --n_samples $NUM_SAMPLES \
        #         --rate_lambda $RATE_LAMBDA \
        #         --workload $WORKLOAD \
        #         --setting $SETTING \
        #         --retraining_rate $RETRAIN_RATE \
        #         --task_assignment "workload" \
        #         --output_dir $OUTPUT_DIR \
        #         --batch_size $BATCH_SIZE \
        #         --alpha $ALPHA \
        #         --beta 0.1 \
        #         --epsilon 0.01 \
        #         --active_selection "adaptive-0.8" \
        #         --k $K \
        #         --experiments $EXPERIMENTS \
        #         --memory_threshold $MEMORY_THRESHOLD

        #     python plot.py \
        #         --node $NUM_NODES \
        #         --model_name $MODEL_NAME \
        #         --setting $SETTING \
        #         --workload $WORKLOAD \
        #         --task_assignment "workload" \
        #         --retraining_rate $RETRAIN_RATE \
        #         --alpha $ALPHA \
        #         --beta 0.1 \
        #         --epsilon 0.01 \
        #         --active_selection "adaptive-0.8" \
        #         --output_dir $OUTPUT_DIR
        # done

        # for BETA in 5.0 1.0 0.5 0.05 0.01; do
        # for BETA in 10.0; do
        #     python distributed_dialogpt.py \
        #         --model_name_or_path "microsoft/$MODEL_NAME" \
        #         --model_name $MODEL_NAME \
        #         --num_nodes $NUM_NODES \
        #         --n_samples $NUM_SAMPLES \
        #         --rate_lambda $RATE_LAMBDA \
        #         --workload $WORKLOAD \
        #         --setting $SETTING \
        #         --retraining_rate $RETRAIN_RATE \
        #         --task_assignment "workload" \
        #         --output_dir $OUTPUT_DIR \
        #         --batch_size $BATCH_SIZE \
        #         --alpha 0.1 \
        #         --beta $BETA \
        #         --epsilon 0.01 \
        #         --active_selection "adaptive-0.8" \
        #         --k $K \
        #         --experiments $EXPERIMENTS \
        #         --memory_threshold $MEMORY_THRESHOLD

        #     python plot.py \
        #         --node $NUM_NODES \
        #         --model_name $MODEL_NAME \
        #         --setting $SETTING \
        #         --workload $WORKLOAD \
        #         --task_assignment "workload" \
        #         --retraining_rate $RETRAIN_RATE \
        #         --alpha 0.1 \
        #         --beta $BETA \
        #         --epsilon 0.01 \
        #         --active_selection "adaptive-0.8" \
        #         --output_dir $OUTPUT_DIR
        # done

    done
done