
# Set NSYS environment variables if needed
stages=6
ngpus=8
microbs=32
nlayers=12
pipeline=pp
export NSYS_NODE_INTERVAL=$((ngpus/stages))
export NSYS_OUTPUT="prof/${pipeline}_transformer-${nlayers}_${stages}stages_${ngpus}gpus_microbs${microbs}"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nsys profile \
#     -f true \
#     -c cudaProfilerApi \
#     -o ${NSYS_OUTPUT} \
#     --trace cuda,nvtx,cudnn,osrt \
#     --export sqlite \
#     python -u pipeline_parallel.py \
#         --nlayers $nlayers \
#         --chunks $microbs \
#         --epochs 1 \
#         --profile

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u pipeline_parallel.py \
    --nlayers $nlayers \
    --chunks $microbs \
    --epochs 1 \
    --profile \
    --do_eval

# --do_train \