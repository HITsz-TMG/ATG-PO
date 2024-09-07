conda activate agent_benchmark
export CUDA_VISIBLE_DEVICES=0
export LLAMA_ROOT=""

directory=""
unbuffer python eval.py --f $directory --citations --gpus 4 | tee -a log_eval.txt
