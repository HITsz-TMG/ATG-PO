# Improving Attributed Text Generation of Large Language Models via Preference Learning

## Overview
Automatic Preference Optimization(APO) framework includes an automatic preference construction procedure and a fine-grained preference optimization procedure, which is used to handle attributed text generation tasks.

## Dataset
The dataset is available [here](https://www.alipan.com/s/21ifsSX229u).

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


## Citation
```bib
@inproceedings{DBLP:conf/acl/LiSHLH0Z24,
  author       = {Dongfang Li and
                  Zetian Sun and
                  Baotian Hu and
                  Zhenyu Liu and
                  Xinshuo Hu and
                  Xuebo Liu and
                  Min Zhang},
  editor       = {Lun{-}Wei Ku and
                  Andre Martins and
                  Vivek Srikumar},
  title        = {Improving Attributed Text Generation of Large Language Models via
                  Preference Learning},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2024,
                  Bangkok, Thailand and virtual meeting, August 11-16, 2024},
  pages        = {5079--5101},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://aclanthology.org/2024.findings-acl.301},
  timestamp    = {Tue, 27 Aug 2024 17:38:11 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/LiSHLH0Z24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}}
```