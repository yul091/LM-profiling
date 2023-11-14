TASK_NAME=mrpc
MINIBATCH=4
STRATEGY=vanilla
MODEL=gpt2

for INPUT_LEN_RATE in 0.5 0.6 0.7 0.8 0.9 1.0; do
  CUDA_VISIBLE_DEVICES=5 python run_glue.py \
    --model_name_or_path $MODEL \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 4 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --logging_strategy steps \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --max_steps 23000 \
    --minibatch $MINIBATCH \
    --strategy $STRATEGY \
    --input_len_rate $INPUT_LEN_RATE \
    --report_to wandb \
    --logging_steps 100 \
    --run_name $TASK_NAME_$MODEL_$STRATEGY-$INPUT_LEN_RATE \
    --output_dir results/$TASK_NAME/$MODEL/$STRATEGY-$INPUT_LEN_RATE/ \
    --overwrite_output_dir
done