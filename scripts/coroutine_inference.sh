
# Set NSYS environment variables if needed
nlayers=64

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
    

CUDA_VISIBLE_DEVICES=4,5,6,7 python -u coroutine_inference.py \
    --nlayers $nlayers \
    --setting random \
    --coroutine \
    --n_samples 50
    