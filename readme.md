# Improving Attributed Text Generation of Large Language Models via Preference Learning


## Environment

```
conda create -n attri_trl python=3.10
conda activate attri_trl
pip install -r requirements.txt
```

## post-training
Using data from `post_training`.

The training scripts is provided in `post_training/scripts`.

`finetune.sh` is used to train base model.

`finetune_selfrag.sh` is used to perform continue post-training after the post-training precedure having finished (ablation study).


## Perference Optimization

Using data from `preference_optimization.jsonl`

The training scripts is provided in `po/scripts`.


## Evaluation

We use the ALCE official evaluation scripts alongwith custom prompts. The evaluation scripts can be found in `evaluation_alce`.