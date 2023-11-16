
# Set NSYS environment variables if needed
nlayers=64
SETTING=random
BATCH=5
N_SAMPLES=$(( 10 * $BATCH ))

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
    

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u coroutine_inference.py \
    --nlayers $nlayers \
    --setting $SETTING \
    --coroutine \
    --bptt $BATCH \
    --n_samples $N_SAMPLES


python plot.py --coroutine --setting $SETTING 