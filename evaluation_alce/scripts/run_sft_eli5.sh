conda activate agent_benchmark
MODEL=""

python run.py \
        --config configs/zeroshot_5docs.yaml \
        --model $MODEL \
        --seed 42 \
        --max_length 4096 \
        --max_new_tokens 1024 \
        --tag llama2_base_bos_eli5_retrieved_sft_step50 \
        --prompt_file prompts/eli5_gpt4_prompt.json \
        --eval_file data/eli5_eval_bm25_top100.json \
        --shot 0 \
        --temperature 0.01 \
        --use_lora True \
        --save_folder result_selfrag_sft
