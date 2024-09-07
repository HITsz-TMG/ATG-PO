conda activate attri_trl
MODEL="" 
DATA=""
OUTPUT_PATH=""

accelerate launch  \
    --config_file="./config/accelerate_config/zero-3.yaml" \
    --main_process_port 25201 \
    --num_processes=2 dpo_train_wo_frab.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_PATH \
    --max_steps 300 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1000 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --logging_steps 20 \
    --model_max_length 4096 \
    --gradient_checkpointing \
    --log_level info \
    --add_bos True \
    --use_lora True
