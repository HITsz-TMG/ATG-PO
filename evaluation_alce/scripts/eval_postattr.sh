conda activate agent_benchmark
export CUDA_VISIBLE_DEVICES=0
export LLAMA_ROOT=""

export DPR_WIKI_TSV=""
export GTR_EMB=""

directory=""
unbuffer python eval_postattr.py --f $directory  | tee -a log_eval.txt
