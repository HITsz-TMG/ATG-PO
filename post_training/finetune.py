import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments
import wandb

# LLAMA_PROMPT = '''<s>[INST] <<SYS>>
# {system}
# <</SYS>>

# {instruction}\n\n[Document]\n{documents}\nQuestion: {Question}\nAnswer: [/INST]'''

VANILLA_PROMPT = '''{instruction}\n\nQuestion: {Question}\n\n{documents}\nAnswer: '''

def load_jsonl(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return [json.loads(l) for l in f]
    
def generate_prompt(inputs, prompt):
    documents = ""
    document_ids = 1
    for item in inputs["documents"]:   
        if not item["text"]:
            continue
        next_document = f'\nDocument [{item["idx"]}](Title: {inputs["title"] if "title"in inputs.keys() else ""}): {item["text"]}\n'
        documents = documents + next_document
        document_ids = document_ids + 1
    instruction = '''Write an accurate, engaging, and concise answer for the given question using only the provided documents (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'''
    return prompt.format_map({
        "instruction": instruction,
        "documents": documents,
        "Question": inputs['query']
    })


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")
    add_special_token: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    add_bos: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    report_to: str = field(default="wandb")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_paths,
        tokenizer,
        model_max_length,
        add_bos
    ):
        super(SupervisedDataset, self).__init__()
        
        self.data = []
        for data_path in data_paths.split(';'):
            self.data = self.data + load_jsonl(data_path)

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        self.add_bos = add_bos
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        if self.add_bos:
            context = self.tokenizer.encode(generate_prompt(example, VANILLA_PROMPT))
        else:
            context = self.tokenizer.encode(generate_prompt(example, VANILLA_PROMPT), add_special_tokens=False)
        answer = self.tokenizer.encode(example["response"], add_special_tokens=False)
        input_ids = context + answer
        labels = [self.ignore_index] * len(context) + answer
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[:self.model_max_length]
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (
            self.model_max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if model_args.add_special_token:
        tokenizer.add_tokens(new_tokens=["<|endofstatement|>", "<|beginofselection|>", "<|endofselection|>"], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.unk_token
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["W_pack"],
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length, add_bos=data_args.add_bos
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()