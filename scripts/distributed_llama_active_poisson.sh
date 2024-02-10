NUM_NODES=2
NUM_SAMPLES=300
WORKLOAD=poisson
SETTING=active
OUTPUT_DIR=prof_async
BATCH_SIZE=2
# RETRAIN_RATE=0.2

for RETRAIN_RATE in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distributed_llama.py \
        --access_token "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" \
        --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
        --num_nodes $NUM_NODES \
        --batch_size $BATCH_SIZE \
        --n_samples $NUM_SAMPLES \
        --workload $WORKLOAD --setting $SETTING \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR

    python plot.py \
        --node $NUM_NODES \
        --model_name "llama2" \
        --setting $SETTING \
        --workload $WORKLOAD \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR
done
