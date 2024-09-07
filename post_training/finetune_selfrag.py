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
import random

VANILLA_PROMPT = '''{instruction}\n\nQuestion: {Question}\n\n{documents}\nAnswer: {finished_answer_span}'''

def load_jsonl(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return [json.loads(l) for l in f]
    
def prepare_mapping(inputs, idx):
    used_documents = []
    for statement in inputs["statements"][:idx+1]:
        if "revised_used_document" in statement.keys():
            used_documents = used_documents + statement["revised_used_document"]
        if "preference" in statement.keys():
            used_documents = used_documents + statement["preference"]["chosen"]["used_document"]
            if "rejeceted_irrelevant_supported" in statement["preference"].keys():
                used_documents = used_documents + statement["preference"]["rejeceted_irrelevant_supported"]["used_document"]
    used_documents = list(set(used_documents))

    missing_element = [i for i in range(1, len(inputs["documents"])+1) if i not in used_documents]
    if len(used_documents) < min(5, len(inputs["documents"])):
        used_documents = used_documents + random.sample(missing_element, min(5 - len(used_documents), len(inputs["documents"]) - len(used_documents)))
    random.shuffle(used_documents)        
    mapping = {}
    for i in range(len(used_documents)):
        mapping[used_documents[i]] = i+1
    return mapping
    
def generate_prompt(inputs, tokenizer, model_max_length, prompt, idx):
    if model_max_length == None:
        model_max_length = 4096
    mapping = prepare_mapping(inputs, idx)

    # prepare documents
    documents = ""
    document_ids = 1
    for key in mapping.keys():
        document = inputs["documents"][int(key)-1]
        if not document["text"]:
            continue
        next_document = f'\nDocument [{document_ids}](Title: {inputs["title"] if "title"in inputs.keys() else ""}): {document["text"]}\n'
        if len(tokenizer.encode(documents+next_document)) > model_max_length:
            break
        documents = documents + next_document
        document_ids = document_ids + 1

    # prepare answer span
    finished_answer_span = []
    for statement in inputs["statements"][:idx]:
        if isinstance(statement["statement"], list):
            statement["statement"] = " ".join(statement["statement"])
        end_sign = ''
        if statement['statement'][-1] in [".", '?', "\"", "!"]:
            end_sign = statement['statement'][-1]
            statement['statement'] = statement['statement'][:-1]
        else:
            end_sign = "."
        if "revised_used_document" in statement.keys():
            finished_answer_span.append(statement["statement"] + "".join(['[{}]'.format(mapping[i]) for i in statement["revised_used_document"]]) + end_sign
            )
        else:
            if statement["used_document"] == []:
                finished_answer_span.append(statement["statement"] + end_sign)
            else:
                indices = "".join(['[{}]'.format(mapping[i]) for i in statement["used_document"]])
                finished_answer_span.append(statement["statement"] + indices + end_sign)
            
    
    finished_answer_span = " ".join(finished_answer_span)
    instruction = '''Write an accurate, engaging, and concise answer for the given question using only the provided documents (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.'''
    return prompt.format_map({
        "instruction": instruction,
        "documents": documents,
        "Question": inputs['query'],
        "finished_answer_span": finished_answer_span
    }), mapping
 

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
        data_path,
        tokenizer,
        model_max_length,
        add_bos
    ):
        super(SupervisedDataset, self).__init__()
        
        dataset = load_jsonl(data_path)
        self.data = []
        for item in dataset:
            for i in range(len(item["statements"])):
                if "preference" not in item["statements"][i].keys():
                    continue
                
                if "rejected_relevant_unsupported" in item['statements'][i]["preference"].keys():
                    prompt, mapping = generate_prompt(item, tokenizer, model_max_length, VANILLA_PROMPT, i)
                    chosen = item['statements'][i]["preference"]["chosen"]["statement"]
                    used_documents = item['statements'][i]["preference"]["chosen"]["used_document"]
                    selection = "".join(['[{}]'.format(mapping[docid]) for docid in used_documents])
                    if len(chosen) == 0:
                        continue

                    if chosen[-1] in [".", ',', '!', '\"', "?"]:
                        end_sign_chosen = chosen[-1]
                        chosen = chosen[:-1]
                    else:
                        end_sign_chosen = '.'

                    if i == len(item["statements"])-1:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection + end_sign_chosen + tokenizer.decode(tokenizer.eos_token_id),
                        })
                    else:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection + end_sign_chosen,
                        })
                if "rejeceted_irrelevant_supported" in item["statements"][i]["preference"].keys():
                    prompt, mapping = generate_prompt(item, tokenizer, model_max_length, VANILLA_PROMPT, i)
                    chosen = item['statements'][i]["preference"]["chosen"]["statement"]
                    used_documents_chosen = item['statements'][i]["preference"]["chosen"]["used_document"]
                    selection_chosen = "".join(['[{}]'.format(mapping[docid]) for docid in used_documents_chosen])
                    if len(chosen) == 0:
                        continue
                    if chosen[-1] in [".", ',', '!', '\"', "?"]:
                        end_sign_chosen = chosen[-1]
                        chosen = chosen[:-1]
                    else:
                        end_sign_chosen = '.'

                    if i == len(item["statements"])-1:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection_chosen + end_sign_chosen + tokenizer.decode(tokenizer.eos_token_id),
                        })
                    else:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection_chosen + end_sign_chosen,
                        })

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
            context = self.tokenizer.encode(example["prompt"])
        else:
            context = self.tokenizer.encode(example["prompt"], add_special_tokens=False)
        answer = self.tokenizer.encode(example["chosen"], add_special_tokens=False)
        input_ids = context + answer
        labels = [self.ignore_index] * len(context) + answer
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

    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

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