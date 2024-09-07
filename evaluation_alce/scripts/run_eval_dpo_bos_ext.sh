conda activate agent_benchmark
MODEL=""

python run.py \
        --config configs/zeroshot_10docs.yaml \
        --model $MODEL \
        --seed 42 \
        --max_length 4096 \
        --max_new_tokens 1024 \
        --tag llama2_base_bos_asqa_retrieved_10docs_step50 \
        --prompt_file prompts/asqa_gpt4_prompt.json \
        --eval_file data/asqa_eval_gtr_top100.json\
        --shot 0 \
        --temperature 0.01 \
        --use_lora True \
        --use_shorter extraction \
        --save_folder result_dpo_extraction
