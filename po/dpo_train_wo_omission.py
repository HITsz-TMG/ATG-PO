from collections import defaultdict
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
from trl.trainer.utils import *

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase

from trl import DPOTrainer

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
    beta: Optional[float] = field(default=0.1)
    loss_type: str = field(default="sigmoid")


class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        add_bos
    ):
        super(DPODataset, self).__init__()

        dataset = load_jsonl(data_path)
        self.data = []
        for item in dataset:
            for i in range(len(item["statements"])):
                if "preference" not in item["statements"][i].keys():
                    continue
                
                if "rejected_relevant_unsupported" in item['statements'][i]["preference"].keys():
                    if item['statements'][i]["preference"]["rejected_relevant_unsupported"]["error_type"] == 'omission':
                        continue
                    prompt, mapping = generate_prompt(item, tokenizer, model_max_length, VANILLA_PROMPT, i)
                    chosen = item['statements'][i]["preference"]["chosen"]["statement"]
                    rejected = item['statements'][i]["preference"]["rejected_relevant_unsupported"]["statement"]
                    used_documents = item['statements'][i]["preference"]["chosen"]["used_document"]
                    selection = "".join(['[{}]'.format(mapping[docid]) for docid in used_documents])
                    if len(chosen) == 0 or len(rejected) == 0:
                        continue

                    if chosen[-1] in [".", ',', '!', '\"', "?"]:
                        end_sign_chosen = chosen[-1]
                        chosen = chosen[:-1]
                    else:
                        end_sign_chosen = '.'

                    if rejected[-1] in [".", ',', '!', '\"', "?"]:
                        end_sign_rejected = rejected[-1]
                        rejected = rejected[:-1]
                    else:
                        end_sign_rejected = '.'

                    if i == len(item["statements"])-1:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection + end_sign_chosen + tokenizer.decode(tokenizer.eos_token_id),
                            "rejected": rejected + selection + end_sign_rejected + tokenizer.decode(tokenizer.eos_token_id)
                        })
                    else:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection + end_sign_chosen,
                            "rejected": rejected + selection + end_sign_rejected
                        })
                if "rejeceted_irrelevant_supported" in item["statements"][i]["preference"].keys():
                    prompt, mapping = generate_prompt(item, tokenizer, model_max_length, VANILLA_PROMPT, i)
                    chosen = item['statements'][i]["preference"]["chosen"]["statement"]
                    rejected = item['statements'][i]["preference"]["rejeceted_irrelevant_supported"]["statement"]
                    used_documents_chosen = item['statements'][i]["preference"]["chosen"]["used_document"]
                    used_documents_rejected = item['statements'][i]["preference"]["rejeceted_irrelevant_supported"]["used_document"]
                    selection_chosen = "".join(['[{}]'.format(mapping[docid]) for docid in used_documents_chosen])
                    selection_rejected = "".join(['[{}]'.format(mapping[docid]) for docid in used_documents_rejected])
                    if len(chosen) == 0 or len(rejected) == 0:
                        continue
                    if chosen[-1] in [".", ',', '!', '\"', "?"]:
                        end_sign_chosen = chosen[-1]
                        chosen = chosen[:-1]
                    else:
                        end_sign_chosen = '.'

                    if rejected[-1] in [".", ',', '!', '\"', "?"]:
                        end_sign_rejected = rejected[-1]
                        rejected = rejected[:-1]
                    else:
                        end_sign_rejected = '.'

                    if i == len(item["statements"])-1:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection_chosen + end_sign_chosen + tokenizer.decode(tokenizer.eos_token_id),
                            "rejected": rejected + selection_rejected + end_sign_rejected + tokenizer.decode(tokenizer.eos_token_id)
                        })
                    else:
                        self.data.append({
                            "prompt": tokenizer.decode(tokenizer.bos_token_id) + prompt if add_bos else prompt,
                            "chosen": chosen + selection_chosen + end_sign_chosen,
                            "rejected": rejected + selection_rejected + end_sign_rejected
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.data[idx]

@dataclass
class CUSTOM_DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id

            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "chosen": chosen_sequence_tokens,
                "rejected": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["rejected_labels"]
                )
                batch["chosen_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["chosen_labels"]
                )

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        # low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16
    )
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        # low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.unk_token
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

    dataset = DPODataset(
        data_args.data_path, tokenizer, training_args.model_max_length, data_args.add_bos
    )
    data_collator = CUSTOM_DPODataCollatorWithPadding(                
                tokenizer,
                max_length=training_args.model_max_length,
                max_prompt_length=training_args.model_max_length,
                label_pad_token_id=-100,
                padding_value=tokenizer.unk_token_id,
                truncation_mode="keep_start",
                is_encoder_decoder=False,
                max_target_length=1024,
        )
    if training_args.remove_unused_columns:
        training_args.remove_unused_columns = False
    trainer = DPOTrainer(
        model=model, ref_model=ref_model,
        beta=training_args.beta,
        max_length=training_args.model_max_length,
        max_prompt_length=training_args.model_max_length,
        data_collator=data_collator,
        args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()