conda activate attri

MASTER_PORT=6114

DATA="" #
DS_CONFIG_PATH="configs/zero-3.json"

MODEL="" # path of llama-2-13b-base
OUTPUT_PATH=""

torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 20 \
    --model_max_length 4096 \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH} \
    --add_bos True \
    --add_special_token True \
    --log_level info
