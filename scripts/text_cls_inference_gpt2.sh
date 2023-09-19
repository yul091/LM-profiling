MODEL=gpt2 # gpt2, bert-base-uncased, facebook/bart-base
EVAL_BATCH=3
OUT_DIR=profile_res

CUDA_VISIBLE_DEVICES=7 python sequence_classification.py \
    --model_name_or_path $MODEL \
    --per_device_eval_batch_size $EVAL_BATCH \
    --output_dir $OUT_DIR 

