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
    # ... (保持不变)

@torch.jit.script
def plausibility_mask(logits: torch.Tensor, alpha: float) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    threshold = log_probs.max(-1, keepdim=True).values + np.log(alpha)
    return log_probs.ge(threshold)

@torch.jit.script
def top_k_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    threshold = logits.topk(k=top_k, dim=-1, largest=True, sorted=True).values[:, :, -1:]
    return logits.ge(threshold)

def build_data(items1, items2, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = len(items1)
    
    assert all(item1["messages"] == item2["messages"] for item1, item2 in zip(items1, items2))
    assert all(item1["labels"] == item2["labels"] for item1, item2 in zip(items1, items2))

    logits1_batch = torch.stack([
        torch.sparse_coo_tensor(
            item["sparse_logits"]["indexes"],
            item["sparse_logits"]["values"],
            item["sparse_logits"]["shape"]
        ).to_dense().to(device)
        for item in items1
    ])

    logits2_batch = torch.stack([
        torch.sparse_coo_tensor(
            item["sparse_logits"]["indexes"],
            item["sparse_logits"]["values"],
            item["sparse_logits"]["shape"]
        ).to_dense().to(device)
        for item in items2
    ])

    if args.sample == "uniform":
        logits_batch = torch.ones_like(logits1_batch)
    elif args.sample == "base":
        logits_batch = logits1_batch
    elif args.sample == "expert":
        logits_batch = logits2_batch
    elif args.sample == "contrast":
        logits_batch = logits1_batch + args.gamma * (logits1_batch - logits2_batch)

    logits_batch.masked_fill_(logits1_batch == 0, torch.finfo(logits_batch.dtype).min)
    logits_batch.masked_fill_(logits2_batch == 0, torch.finfo(logits_batch.dtype).min)

    samples_batch = []
    for batch_idx in range(batch_size):
        samples = []
        labels = items1[batch_idx]["labels"]
        for idx in range(logits_batch.shape[1]):
            label = labels[idx]
            if label == -100:
                continue
            token_sample = torch.multinomial(
                F.softmax(logits_batch[batch_idx, idx] / args.temperature, -1),
                num_samples=args.num_samples,
                replacement=True
            ).tolist()
            for token in token_sample:
                if logits_batch[batch_idx, idx, token] == torch.finfo(logits_batch.dtype).min:
                    continue
                assert token != label
                samples.append((idx+1, label, token))
        samples_batch.append({"messages": items1[batch_idx]["messages"], "samples": samples})

    return samples_batch

@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds1 = datasets.load_dataset(args.dataset1)["train"].sort("idx")
    ds2 = datasets.load_dataset(args.dataset2)["train"].sort("idx")
    assert len(ds1) == len(ds2)
    
    batch_size = 32  # 可以根据GPU内存调整
    
    corpus = []
    for idx in tqdm(range(0, len(ds1), batch_size)):
        ds1_items = ds1.select(list(range(idx, min(idx + batch_size, len(ds1)))))
        ds2_items = ds2.select(list(range(idx, min(idx + batch_size, len(ds1)))))
        
        corpus.extend(build_data(ds1_items, ds2_items, args))
    
    dataset = datasets.DatasetDict()
    dataset["train"] = datasets.Dataset.from_list(corpus)
    dataset.save_to_disk(args.save_dir)

if __name__ == '__main__':    
    main()