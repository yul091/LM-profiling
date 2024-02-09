NUM_NODES=2
NUM_SAMPLES=5000
LAYERS=12
WORKLOAD=all
SETTING=random
OUTPUT_DIR=prof_async

for RETRAIN_RATE in 0 0.2 0.4 0.6 0.8 1.0; do

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distributed_transformer.py \
        --num_nodes $NUM_NODES \
        --n_samples $NUM_SAMPLES \
        --nlayers $LAYERS \
        --emsize 768 \
        --nhead 8 \
        --nhid 768 \
        --batch_size 16 \
        --workload $WORKLOAD \
        --setting $SETTING \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR

    python plot.py \
        --node $NUM_NODES \
        --coroutine \
        --setting $SETTING \
        --workload $WORKLOAD \
        --retraining_rate $RETRAIN_RATE \
        --output_dir $OUTPUT_DIR

done
