import os
import re
import sys
import json
import time
import types
import random
import argparse
import shortuuid
import importlib
from tqdm import tqdm
from functools import partial
from typing import List, Dict
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_dir", type=str, default=None)
    
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--expert_model_name_or_path", type=str, default=None)
    parser.add_argument("--anti_expert_model_name_or_path", type=str, default=None)
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8bit')
    
    parser.add_argument("--max_length", type=int, default=None)    
    
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--micro_batch_size_per_gpu', type=int, default=1)

    return parser.parse_args()

def load_fn(args):
    
    with open("./data/evaluation/mt_bench/question.jsonl", "r") as ques_file:      
        questions = [json.loads(line) for line in ques_file if line]
    questions = sorted(questions, key=lambda x: int(x["question_id"]))
    
    with open("./data/evaluation/mt_bench/answer.jsonl", "r") as ans_file:      
        responses = [json.loads(line) for line in ans_file if line]
    responses = sorted(responses, key=lambda x: int(x["question_id"]))
    
    datasets = []
    for question, response in zip(questions, responses):
        datasets.append([
            {"role": "user", "content": question["turn"][0]},
            {"role": "assistant", "content": response["choices"][0]["turns"][0]},
            {"role": "user", "content": question["turn"][1]},
            {"role": "assistant", "content": response["choices"][0]["turns"][1]},
        ])
    
    return datasets

@torch.inference_mode()
def main():

    start = time.time()
    args = parse_args()
    
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                             torch_dtype=torch.float16, 
                                             device_map=local_rank, 
                                             load_in_8bit=args.load_in_8bit, 
                                             low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
    if args.expert_model_name_or_path is not None:
        expert_model = AutoModelForCausalLM.from_pretrained(args.expert_model_name_or_path, 
                                                            torch_dtype=torch.float16, 
                                                            device_map=local_rank, 
                                                            load_in_8bit=args.load_in_8bit, 
                                                            low_cpu_mem_usage=True)
        if len(tokenizer) != expert_model.config.vocab_size:
            expert_model.resize_token_embeddings(len(tokenizer))
    
    for param in model.parameters():
        param.requires_grad = False

    dataset = load_fn(args)

    results = []
    for messages in tqdm(dataset):
        messages = []
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        
        logits = model(input_ids.to(local_rank)).logits[:, :-1]
        expert_logits = expert_model(input_ids.to(local_rank)).logits[:, :-1]
        
        filtering_mask = ~(logits.argmax(-1).eq(expert_logits.argmax(-1)))
        
        
        
        # find the different position
        for i in range(logits.shape[0]):
            if logits[i].argmax() == expert_logits[i].argmax():
                results.append({"question_id": len(results), "answer": "correct"})
        
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"results_{local_rank}.jsonl"), "w") as fout:
        fout.writelines([json.dumps(example) + "\n" for example in results])
            
    accelerator.wait_for_everyone()  
    if accelerator.is_local_main_process:
        results = []
        for rank in range(accelerator.num_processes):
            file_path = os.path.join(args.save_dir, f"results_{rank}.jsonl")
            with open(file_path) as fin:
                results.extend([json.loads(line) for line in fin])
            os.remove(file_path)
        # remove duplicates
        results = {result['question_id']: result for result in results}
        results = list({key: results[key] for key in sorted(list(results.keys()), key=lambda x: int(x))}.values())
                
        with open(os.path.join(args.save_dir, "results.jsonl"), "w") as fout:
            fout.writelines([json.dumps(example) + "\n" for example in results])
        with open(os.path.join("../FastChat/fastchat/llm_judge/data/mt_bench/model_answer", f"{args.model_id}.jsonl"), "w") as fout:
            fout.writelines([json.dumps(example) + "\n" for example in results])            
        
        print(f"Inference on MT-Bench is done!\n")
            
if __name__ == "__main__":

    main()