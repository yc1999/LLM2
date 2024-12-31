import os
import re
import sys
import json
import time
import random
import argparse
import importlib
from tqdm import tqdm
from functools import partial
from typing import List, Dict
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import PeftModel, PeftConfig
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer

from verifier import ProcessVerifier
from verify_search import LLM2

from utils_stopping import StopAtSpecificTokenCriteria
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
PROMPT_FORMAT = {
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
    "qwen": '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
}

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--max_num_examples_per_task", type=int, default=None)
    
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--verifier_model_name_or_path", type=str, default=None)
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8bit')
    
    parser.add_argument("--no_cot", action="store_true")
    parser.add_argument("--n_shot", type=int, default=0)
    parser.add_argument("--prompt_format", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--eval_pass_at_ks", nargs="+", type=int, default=[1], help="Multiple k's that we will report pass@k.")
    
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=None)
    
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--micro_batch_size_per_gpu', type=int, default=1)
    parser.add_argument('--preprocessing_num_workers', type=int, default=4)
    parser.add_argument('--writer_batch_size', type=int, default=1000)
    parser.add_argument('--keep_in_memory', action='store_true', help='Keep in memory')
    parser.add_argument('--overwrite_cache', action='store_true', help='Overwrite the cached training and evaluation sets')

    parser.add_argument('--do_sample', action='store_true', help="Whether to do sampling")

    return parser.parse_args()

def generate(batch, args, model, tokenizer):
    
    model.generation_config.update(**{
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    })
    
    stopping_criteria = StoppingCriteriaList()
    if not args.use_chat_format:
        stopping_criteria.append(
            StopAtSpecificTokenCriteria(token_id_list=[[271], [382]])
        )
    batch_size = batch["input_ids"].shape[0]

    start_time = time.time()

    outputs = model.generate(
        **{key: value for key, value in batch.items() if torch.is_tensor(value)},
        generation_config=model.generation_config, stopping_criteria=stopping_criteria,
    )

    end_time = time.time()
    generation_time = end_time - start_time

    # we need to re-encode the prompt because some tokenizers won't add space token before the first token
    batch_prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    batch_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    batch["prediction"] = [output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)]

    results = [{key: batch[key][idx] for key in batch.keys() if not torch.is_tensor(batch[key])} for idx in range(batch_size)]
    for result in results:
        result['generation_time'] = generation_time / batch_size  # 每个样本的平均生成时间
    return results
    # return [{key: batch[key][idx] for key in batch.keys() if not torch.is_tensor(batch[key])} for idx in range(batch_size)]

def discriminate(batch, args, model, tokenizer):
    
    batch_size = len(batch["prompt"])
    input_texts = [f"{batch['prompt'][0]} {choice}{tokenizer.eos_token}" for choice in batch["choices"][0]]

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = tokenizer(input_texts, padding="longest", return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(batch["input_ids"].device)
    
    outputs = model(input_ids=input_ids, return_dict=True)
    logits = outputs.logits[:, batch["input_ids"].shape[-1]-1: -1, :]
    log_probs = logits.log_softmax(-1).view(-1, logits.shape[-1])
    scores = log_probs[torch.arange(log_probs.shape[0], device=logits.device), 
                       input_ids[:, batch["input_ids"].shape[-1]:].reshape(-1)].view(input_ids.shape[0], -1)

    if hasattr(model, "verifier"):
        verifier_outputs = model.verifier(input_ids=input_ids, return_dict=True)
        verifier_scores = verifier_outputs.logits[:, batch["input_ids"].shape[-1]:]  
        scores = scores + args.beta * verifier_scores

    sequence_lengths = (torch.eq(input_ids[:, batch["input_ids"].shape[-1]:], tokenizer.eos_token_id).int().argmax(-1)).to(logits.device)
    scores = scores.cumsum(dim=-1)[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths-1]
    
    batch["scores"] = [scores.tolist()]
    batch["scores_norm"] = [[score / len(batch["choices"][0][idx]) for idx, score in enumerate(scores.tolist())]]
    
    return [{key: batch[key][idx] for key in batch.keys() if not torch.is_tensor(batch[key])} for idx in range(batch_size)]

def collate_fn(batch: List[Dict[str, torch.Tensor]], args, tokenizer):

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    encoded = tokenizer([x["prompt"] for x in batch], 
                        padding="longest", 
                        truncation=True,
                        max_length=args.max_input_tokens,
                        return_tensors="pt", add_special_tokens=False)
    
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        **{key: [x[key] for x in batch] for key in batch[0].keys()}
    }

@torch.inference_mode()
def main():

    start = time.time()
    args = parse_args()
    
    benchmark_module = importlib.import_module(f"benchmarks.{args.benchmark}")
    load_fn = benchmark_module.load
    eval_fn = benchmark_module.eval
    if benchmark_module.TASK_TYPE == "generation":
        inference_fn = generate
    elif benchmark_module.TASK_TYPE == "discrimination":
        inference_fn = discriminate
        
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
    if args.verifier_model_name_or_path is not None:
        verifier = ProcessVerifier.from_pretrained(args.verifier_model_name_or_path, 
                                                   num_labels=1, 
                                                   torch_dtype=torch.float16,
                                                   device_map=local_rank, 
                                                   load_in_8bit=args.load_in_8bit, 
                                                   low_cpu_mem_usage=True)
        if len(tokenizer) != verifier.config.vocab_size:
            verifier.resize_token_embeddings(len(tokenizer))        
        model = LLM2(model, verifier, top_k=args.top_k, alpha=args.alpha, beta=args.beta)
        assert args.micro_batch_size_per_gpu == 1, "Micro batch size must be 1 for verification."
        
    for param in model.parameters():
        param.requires_grad = False

    if args.prompt_format is not None:
        prompt_format = PROMPT_FORMAT[args.prompt_format]
        print(f"Using prompt format: {prompt_format}")
    else:
        prompt_format = None
        print("No prompt format is used.")

    dataset = load_fn(args, prompt_format)

    with accelerator.main_process_first():
        dataset = dataset.map(
                    lambda x: {"prompt": x["prompt"] + args.prefix}, 
                    num_proc=args.preprocessing_num_workers,
                    writer_batch_size=args.writer_batch_size,
                    keep_in_memory=args.keep_in_memory,
                    load_from_cache_file=not args.overwrite_cache,)
    accelerator.wait_for_everyone()
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer),
                                             num_workers=args.num_workers,
                                             batch_size=args.micro_batch_size_per_gpu,  
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,)

    dataloader = accelerator.prepare(dataloader)

    results = []
    total_generation_time = 0
    total_samples = 0
    for batch in tqdm(dataloader):
        # results.extend(inference_fn(batch=batch, args=args, model=model, tokenizer=tokenizer))
        batch_results = inference_fn(batch=batch, args=args, model=model, tokenizer=tokenizer)
        results.extend(batch_results)
        total_generation_time += sum(result['generation_time'] for result in batch_results)
        total_samples += len(batch_results)

    average_generation_time = total_generation_time / total_samples

    total_generation_time += sum(result['generation_time'] for result in batch_results)
    total_samples += len(batch_results)
        
    os.makedirs(args.save_dir, exist_ok=True)
    global_rank = accelerator.process_index
    with open(os.path.join(args.save_dir, f"results_{global_rank}.jsonl"), "w") as fout:
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
        results = {result['idx']: result for result in results}
        results = list({key: results[key] for key in sorted(list(results.keys()), key=lambda x: int(x))}.values())
        
        metrics, results = eval_fn(args, results)
        metrics["time"] = f"{time.time() - start:.2f} seconds"
        print(f"Average generation time per sample: {average_generation_time:.4f} seconds")
        metrics["average_generation_time"] = average_generation_time
        print(metrics)
                
        json.dump(metrics, open(os.path.join(args.save_dir, "metrics.json"), "w"), indent=4)
        with open(os.path.join(args.save_dir, "results.jsonl"), "w") as fout:
            fout.writelines([json.dumps(example) + "\n" for example in results])
        with open(os.path.join(args.save_dir, "args.txt"), "w") as fout:
            fout.write("\n".join(f"{arg}: {value}" for arg, value in vars(args).items()))
        
        print(f"Evalution on {args.benchmark} is done!\n")
            
if __name__ == "__main__":

    main()