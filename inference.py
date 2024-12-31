import os
import re
import sys
import copy
import json
import time
import torch
import openai
import random
import shutil
import argparse
import numpy as np
import transformers
from tqdm import tqdm
from functools import partial
from datetime import timedelta
from scipy.sparse import csr_matrix

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Iterable

import datasets
from datasets import load_dataset
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# DEFAULT_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
DEFAULT_CHAT_TEMPLATE = "{{ bos_token }}{% set system_messages = messages | selectattr('role', 'equalto', 'system') | selectattr('content', 'ne', '') | list %}{% if system_messages | length > 0 %}{% for message in system_messages %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + eos_token }}{% endfor %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}{% endfor %}"

def find_sublist(main_list, sublist):
    sublist_len = len(sublist)
    start_index = next((i for i in range(len(main_list) - sublist_len + 1) if main_list[i:i + sublist_len] == sublist), None)
    if start_index is not None:
        return (start_index, start_index + sublist_len)
    return None

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_split", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true")
    
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    
    parser.add_argument('--load_from_disk', action='store_true')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--preprocessing_num_workers', type=int)
    parser.add_argument('--keep_in_memory', action='store_true', help='Keep in memory')
    parser.add_argument('--load_from_cache', action='store_true')
    parser.add_argument('--per_device_train_batch_size', type=int)
    parser.add_argument('--save_steps', type=int)

    return parser.parse_args()

def plausibility_mask(logits: torch.FloatTensor, alpha) -> torch.FloatTensor:
    
    log_probs = logits.log_softmax(dim=-1)
    threshold = log_probs.max(-1, keepdim=True).values + np.log(alpha)
    
    return log_probs.ge(threshold)

def top_k_mask(logits: torch.FloatTensor, top_k) -> torch.FloatTensor:
    
    threshold = logits.topk(k=top_k, dim=-1, largest=True, sorted=True).values[:, :, -1:]
    
    return logits.ge(threshold)

@torch.no_grad()
def main():
    
    start = time.time()
    args = parse_args()

    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # set chat template
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    print("Chat Template: ", tokenizer.chat_template)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    print("Pad Token ID: ", tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                 torch_dtype=torch.float16, 
                                                 device_map=local_rank, 
                                                 load_in_8bit=args.load_in_8bit, 
                                                 low_cpu_mem_usage=True) 
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))  

    print("Loading dataset...")
    if args.load_from_disk:
        if args.data_split != None:
            dataset = datasets.load_from_disk(os.path.join(args.dataset, args.data_split), 
                                          keep_in_memory=args.keep_in_memory)
        else:
            dataset = datasets.load_from_disk(args.dataset, 
                                          keep_in_memory=args.keep_in_memory)
    else:
        if args.data_split != None:
            dataset = datasets.load_dataset(args.dataset, 
                                        split=args.data_split, 
                                        num_proc=args.preprocessing_num_workers,
                                        keep_in_memory=args.keep_in_memory)
        else:
            dataset = datasets.load_dataset(args.dataset,
                                        num_proc=args.preprocessing_num_workers,
                                        keep_in_memory=args.keep_in_memory)            
    
    def process_fn(example):
        print("<<Bug>> Example", example)
        input_ids = tokenizer.apply_chat_template(example["messages"], tokenize=True)
        
        labels = [x for x in input_ids]
        for turn in example["messages"]:
            if turn["role"] == "user":
                turn_input_ids = tokenizer.apply_chat_template([turn], tokenize=True)
                (start, end) = find_sublist(input_ids, turn_input_ids)
                assert input_ids[start: end] == turn_input_ids
                labels[start: end] = [-100] * len(turn_input_ids)

        input_ids = input_ids[:args.max_length]
        labels = labels[:args.max_length]
                                        
        return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}

    def collate_fn(batch: List[Dict[str, Any]]):
        
        input_ids = tokenizer.pad(
            [{"input_ids": example["input_ids"]} for example in batch],
            return_tensors="pt",
            padding=True,
        )["input_ids"]
        
        labels = torch.nn.utils.rnn.pad_sequence(
            [example["labels"] for example in batch], 
            batch_first=True, 
            padding_value=-100
        )
         
        return {
            "messages": [example["messages"] for example in batch],
            "input_ids": input_ids,
            "labels": labels,
            "length": [example["length"] for example in batch],
        }

    with accelerator.main_process_first():
        dataset = dataset.map(process_fn, 
                              num_proc=args.preprocessing_num_workers, 
                              keep_in_memory=args.keep_in_memory,
                              load_from_cache_file=args.load_from_cache,
                              desc="Processing Data").with_format("torch")
    accelerator.wait_for_everyone()
    
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             collate_fn=collate_fn,
                                             batch_size=args.per_device_train_batch_size, 
                                             num_workers=args.num_workers, 
                                             pin_memory=True, 
                                             shuffle=False,
                                             drop_last=False)     

    dataloader = accelerator.prepare(dataloader)
    
    def get_cache_dir(rank=None):
        cache_pattern = re.compile(r"step_(\d+)_to_(\d+)_rank_(\d+)")
        if rank is None:
            return [os.path.join(args.save_dir, dir) for dir in os.listdir(args.save_dir) if cache_pattern.match(dir)]
        else:
            return [os.path.join(args.save_dir, dir) for dir in os.listdir(args.save_dir) \
                    if cache_pattern.match(dir) and cache_pattern.match(dir).group(3) == str(rank)]
        
    cache_dir = get_cache_dir(rank=local_rank)    
    if len(cache_dir) > 0:
        steps = [int(re.search(r"step_(\d+)_to_(\d+)_rank_(\d+)", dir).group(2)) for dir in cache_dir]
        finished_steps = max(steps)
        print(f"Found cache at step {finished_steps}!")
    else:
        finished_steps = 0
    
    started_steps = finished_steps + 1        
    corpus = []
    for step, batch in enumerate(tqdm(dataloader)):
        if step + 1 <= finished_steps:
            continue
        
        input_ids = batch["input_ids"]
        logits = model(input_ids).logits
        if args.alpha > 0 and args.top_k > 0:
            filtering_mask = plausibility_mask(logits, args.alpha) & top_k_mask(logits, args.top_k)
        elif args.alpha > 0:
            filtering_mask = plausibility_mask(logits, args.alpha)
        elif args.top_k > 0:
            filtering_mask = top_k_mask(logits, args.top_k)
        filtering_mask[:, :, tokenizer.pad_token_id] = False
        logits.masked_fill_(~filtering_mask, 0)

        shift_logits = logits[:, :-1, :]
        shift_lables = input_ids[:, 1:]
        logits_mask = torch.arange(logits.shape[-1], device=logits.device)[None, None, :].expand(shift_logits.shape)
        masked_shift_logits = shift_logits.masked_fill(logits_mask == shift_lables[:, :, None], torch.finfo(logits.dtype).min)                
        
        logits = masked_shift_logits
        filtering_mask = filtering_mask[:, :-1, :]
        
        for id1 in range(input_ids.shape[0]):
            
            indexes = torch.nonzero(filtering_mask[id1, :batch["length"][id1]], as_tuple=True)
            sparse_indexes = torch.cat([index.unsqueeze(0) for index in indexes], dim=0)
            
            # sparse_logits = torch.sparse_coo_tensor(sparse_indexes, logits[id1, :batch["length"][id1]][indexes], logits[id1, :batch["length"][id1]].shape)
            # sparse_to_dense_logits = sparse_logits.to_dense()
            # assert torch.equal(logits[id1, :batch["length"][id1]].float(), sparse_to_dense_logits.float())
            corpus.append({
                "idx": step * args.per_device_train_batch_size * accelerator.num_processes + id1 * accelerator.num_processes + local_rank,
                "messages": batch["messages"][id1],
                "sparse_logits": {
                    "indexes": sparse_indexes.cpu(),
                    "values": logits[id1][indexes].cpu(),
                    "shape": logits[id1].shape
                },
                "labels": batch["labels"][id1, 1:batch["length"][id1]].cpu(),
                })
            
        if (args.save_steps and (step + 1) % args.save_steps == 0) or step == len(dataloader) - 1:
            dataset = datasets.Dataset.from_list(corpus)
            dataset.save_to_disk(os.path.join(args.save_dir, f"step_{started_steps}_to_{step+1}_rank_{local_rank}"))
            started_steps = step + 2
            corpus = []

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cache_dir = get_cache_dir()
        total_shards = 0
        for dir in cache_dir:
            arrow_files = [file for file in os.listdir(dir) if file.endswith(".arrow")]
            total_shards += len(arrow_files)
            
        current_shard_index = 0 
        for dir in cache_dir:
            arrow_files = [file for file in os.listdir(dir) if file.endswith(".arrow")]
            for file in arrow_files:
                new_filename = f"data-{current_shard_index:05d}-of-{total_shards:05d}.arrow"
                current_shard_index += 1

                source_file_path = os.path.join(dir, file)
                destination_file_path = os.path.join(args.save_dir, new_filename)
                shutil.move(source_file_path, destination_file_path)
            shutil.rmtree(dir)

        print("Done!")
        
if __name__ == '__main__':    

    main()   

