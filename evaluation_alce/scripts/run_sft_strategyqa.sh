conda activate agent_benchmark
MODEL=""
python run.py \
        --config configs/zeroshot_5docs.yaml \
        --model $MODEL \
        --seed 42 \
        --max_length 4096 \
        --max_new_tokens 1024 \
        --tag llama2_base_bos_strategyqa_retrieved \
        --prompt_file prompts/asqa_gpt4_prompt.json \
        --eval_file data/strategyqa_eval_gtr_top100.json\
        --shot 0 \
        --temperature 0.01 \
        --save_folder result_sft
