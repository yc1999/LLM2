# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, List, Literal, Optional, Union

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from configs import DataArguments

def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task in ["dpo", "orpo"]:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False


def get_datasets(
    data_config: Union[DataArguments,dict],
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        data_config,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
    )
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict,
    args: Union[DataArguments,dict],
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            if args.load_from_disk:
                dataset = load_from_disk(os.path.join(ds, split))
            else:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets

def process_ultrafeedback_binarized():
        
    dataset = load_dataset("../../DATASET/HuggingFaceH4/ultrafeedback_binarized")
    
    chosen_dataset = DatasetDict()
    rejected_dataset = DatasetDict()
    for split in ["train_prefs", "test_prefs"]:
        
        ds = dataset[split]
        target_split = split.split("_")[0]
        chosen_dataset[target_split] = ds.map(lambda example: {
            "messages": example["chosen"]
        }, remove_columns=[col for col in ds.column_names if col != "messages"])
        rejected_dataset[target_split] = ds.map(lambda example: {
            "messages": example["rejected"]
        }, remove_columns=[col for col in ds.column_names if col != "messages"])
        
    print("ultrafeedback_binarized", len(chosen_dataset), len(rejected_dataset))
    
    chosen_dataset.save_to_disk("./data/ultrafeedback_binarized_chosen")
    rejected_dataset.save_to_disk("./data/ultrafeedback_binarized_rejected")
    
    return dataset

def process_deita_10k():
            
    def process(example):
        messages = [
            {"role": "user", "content": turn["value"]} if turn["from"] == "human" else {"role": "assistant", "content": turn["value"]} for turn in example["conversations"]
        ]
        if messages[0]["role"] == "assistant":
            messages = messages[1:]
        return {"messages": messages}
            
    dataset = load_dataset("../../DATASET/hkust-nlp/deita-10k-v0")
    dataset = dataset.map(process, 
                          remove_columns=[col for col in dataset.column_names["train"] if col != "messages"])
    
    print(dataset)
    dataset.save_to_disk("./data/hkust-nlp/deita-10k-v0")
    
    return dataset

def process_mathinstruct():
    
    def process(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        return {"messages": messages}
    
    dataset = load_dataset("./DATASET/TIGER-Lab/MathInstruct")
    # print(dataset.column_names["train"])
    dataset = dataset.map(process,
                          remove_columns=[col for col in dataset.column_names["train"] if col != "messages"])
    
    print(dataset.column_names["train"])
    # print(dataset["train"][0])
    dataset.save_to_disk("./data/TIGER-Lab/MathInstruct")

def process_openmathinstruct():
    
    def process(example):
        messages = [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["generated_solution"]}
        ]
        return {"messages": messages}
    
    dataset = load_dataset("./DATASET/nvidia/OpenMathInstruct-2")
    # print(dataset.column_names["train"])
    dataset = dataset.map(process,
                          remove_columns=[col for col in dataset.column_names["train"] if col != "messages"])
    
    print(dataset.column_names["train"])
    # print(dataset["train"][0])
    dataset.save_to_disk("./data/nvidia/OpenMathInstruct-2")

def process_gsm8k():

    def process(example):
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["cot_path"]}
        ]
        return {"messages": messages}
    
    dataset = load_dataset('json', data_files='./DATASET/gsm8k/gsm8k.jsonl')
    dataset = dataset.map(process,
                          remove_columns=[col for col in dataset.column_names["train"] if col != "messages"])
    
    print(dataset.column_names["train"])
    print(dataset["train"][0])
    dataset.save_to_disk("./data/gsm8k")

def process_math():

    def process(example):
        messages = [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]}            
        ]
        return {"messages": messages}
    
    dataset = load_dataset('./DATASET/lighteval/MATH')

    dataset = dataset.map(process,
                          remove_columns=[col for col in dataset.column_names["train"] if col != "messages"])
    print(dataset.column_names["train"])
    dataset.save_to_disk("./data/lighteval/MATH")


def process_synthetic_gsm8k():

    def process(example):
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["cot_path"]}
        ]
        return {"messages": messages}
    
    dataset = load_from_disk("./DATASET/gsm8k/Meta-Llama-3.1-8B-Instruct")
    print(dataset)
    dataset = dataset.map(process,
                          remove_columns=[col for col in dataset.column_names["train"] if col != "messages"])
    
    print(dataset.column_names["train"])
    print(dataset["train"][0])
    dataset.save_to_disk("./data/gsm8k/Meta-Llama-3.1-8B-Instruct")

def process_synthetic_math(model_name):

    def process(example):
        messages = [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]}
        ]
        return {"messages": messages}
    
    dataset = load_from_disk(f"./DATASET/math/{model_name}")
    print(dataset)
    dataset = dataset.map(process,
                          remove_columns=[col for col in dataset.column_names["train"] if col != "messages"])
    
    print(dataset.column_names["train"])
    print(dataset["train"][0])
    dataset.save_to_disk(f"./data/math/{model_name}")

if __name__ == "__main__":

    process_synthetic_math("Meta-Llama-3.1-8B-Instruct")