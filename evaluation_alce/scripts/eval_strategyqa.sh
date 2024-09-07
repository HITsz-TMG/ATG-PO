conda activate agent_benchmark
directory=""
unbuffer python eval_strategyqa.py --f $directory --citations --gpus 0 | tee -a log_eval.txt
