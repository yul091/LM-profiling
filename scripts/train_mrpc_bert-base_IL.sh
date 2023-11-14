TASK_NAME=mrpc
MODEL=bert-base-cased

# # FIRST train a IL model with validation dataset
# CUDA_VISIBLE_DEVICES=0 python run_glue_IL.py \
#   --model_name_or_path $MODEL \
#   --task_name $TASK_NAME \
#   --do_train \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --evaluation_strategy steps \
#   --save_strategy steps \
#   --load_best_model_at_end \
#   --metric_for_best_model loss \
#   --logging_strategy steps \
#   --save_total_limit 1 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 100 \
#   --strategy all \
#   --logging_steps 500 \
#   --output_dir results/$TASK_NAME/$MODEL/$STRATEGY/IL


MINIBATCH=4
STRATEGY=IL

# SECOND train a model with train dataset
CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path $MODEL \
  --IL_model_path results/$TASK_NAME/$MODEL/all/IL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --load_best_model_at_end \
  --metric_for_best_model accuracy \
  --logging_strategy steps \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 100 \
  --minibatch $MINIBATCH \
  --strategy $STRATEGY \
  --logging_steps 100 \
  --report_to wandb \
  --run_name run_glue_mrpc_$MODEL_$STRATEGY \
  --output_dir results/$TASK_NAME/$MODEL/$STRATEGY \
  --overwrite_output_dir