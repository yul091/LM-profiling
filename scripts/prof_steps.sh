#!/bin/bash -l
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=g34
#SBATCH --output=interleave6.txt

# module load daint-gpu
# conda activate py38_kfac
# export MASTER_ADDR=$(hostname)

#model=bert-base
model=transformer
pipeline='gpipe'
#pipeline='1f1b'
#pipeline='chimera'
# pipeline='interleave'
stages=3
ngpus=3
microbs=8
# acc=1
nlayers=12
export NSYS_NODE_INTERVAL=$((ngpus/stages))
export NSYS_OUTPUT=profile/${model}-${nlayers}_${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}

CUDA_VISIBLE_DEVICES=2,4,5 srun --wait=0 scripts/nsys_wrap.sh \
    python pipeline_parallel.py \
        --nlayers ${nlayers} \
	    --chunks ${microbs} 
