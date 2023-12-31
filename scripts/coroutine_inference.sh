
# Set NSYS environment variables if needed
nlayers=64
SETTING=random
BATCH=5
N_SAMPLES=$(( 10 * $BATCH ))
OUTPUT_DIR=prof_new

# for SETTING in identical random increasing decreasing; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python -u coroutine_inference.py \
#     --nlayers $nlayers \
#     --setting $SETTING \
#     --coroutine \
#     --profiling
# done

# for SETTING in identical random increasing decreasing; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python -u coroutine_inference.py \
#     --nlayers $nlayers \
#     --setting $SETTING \
#     --profiling
# done
    

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u coroutine_inference.py \
    --nlayers $nlayers \
    --setting $SETTING \
    --coroutine \
    --bptt $BATCH \
    --output_dir $OUTPUT_DIR \
    --n_samples $N_SAMPLES


python plot.py --coroutine --setting $SETTING --output_dir $OUTPUT_DIR