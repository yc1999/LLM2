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
import multiprocessing
from tqdm import tqdm
from functools import partial
from datetime import timedelta

import torch.nn as nn
import torch.nn.functional as F

import datasets
from accelerate.utils import set_seed

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_split", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--dataset1", type=str, default=None)
    parser.add_argument("--dataset2", type=str, default=None)
    
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None) 
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--preprocessing_num_workers', type=int)

    return parser.parse_args()

def plausibility_mask(logits: torch.FloatTensor, alpha) -> torch.FloatTensor:
    
    log_probs = logits.log_softmax(dim=-1)
    threshold = log_probs.max(-1, keepdim=True).values + np.log(alpha)
    
    return log_probs.ge(threshold)

def top_k_mask(logits: torch.FloatTensor, top_k) -> torch.FloatTensor:
    
    threshold = logits.topk(k=top_k, dim=-1, largest=True, sorted=True).values[:, :, -1:]
    
    return logits.ge(threshold)

def build_data(item1, item2, args):
    assert item1["messages"] == item2["messages"]
    assert item1["labels"] == item2["labels"]

    logits1 = torch.sparse_coo_tensor(
        item1["sparse_logits"]["indexes"], item1["sparse_logits"]["values"], item1["sparse_logits"]["shape"]).to_dense()
    logits2 = torch.sparse_coo_tensor(
        item2["sparse_logits"]["indexes"], item2["sparse_logits"]["values"], item2["sparse_logits"]["shape"]).to_dense()
        
    if args.sample == "uniform":
        logits = torch.ones_like(logits1)
    elif args.sample == "base":
        logits = logits1
    elif args.sample == "expert":
        logits = logits2
    elif args.sample == "contrast":
        logits = logits1 + args.gamma * (logits1 - logits2)
        
    logits.masked_fill_(logits1 == 0, torch.finfo(logits.dtype).min)
    logits.masked_fill_(logits2 == 0, torch.finfo(logits.dtype).min)
        
    samples = []
    for idx in range(logits.shape[0]): 
        label = item1["labels"][idx]
        if label == -100:
            continue
        token_sample = torch.multinomial(
            F.softmax(logits[idx] / args.temperature, -1), num_samples=args.num_samples, replacement=True).tolist()
        for token in token_sample:
            if logits[idx][token] == torch.finfo(logits.dtype).min:
                continue
            assert token != label     
            samples.append((idx+1, label, token))
    
    return {"messages": item1["messages"], "samples": samples}

@torch.no_grad()
def main():
    
    args = parse_args()
    set_seed(args.seed)
    
    ds1 = datasets.load_dataset(args.dataset1)["train"].sort("idx")
    ds2 = datasets.load_dataset(args.dataset2)["train"].sort("idx")
    assert len(ds1) == len(ds2)
    
    pool = multiprocessing.Pool(args.num_workers)
    corpus = []
    for idx in tqdm(range(0, len(ds1), args.num_workers)):
        ds1_items = ds1.select(list(range(idx, idx + args.num_workers)))
        ds2_items = ds2.select(list(range(idx, idx + args.num_workers)))
        
        corpus.extend(
            pool.starmap(partial(build_data, args=args), list(zip(ds1_items, ds2_items))))
    
    pool.close()
    pool.join()
                    
    dataset = datasets.DatasetDict()
    dataset["train"] = datasets.Dataset.from_list(corpus)
    dataset.save_to_disk(args.save_dir)
        
if __name__ == '__main__':    

    main()   