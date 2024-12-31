import os
import re
import sys
import json
import random
import argparse
from tqdm import tqdm
from functools import partial

import torch
import evaluate
from datasets import Dataset

import benchmarks.utils as utils

TASK_TYPE = "generation"
    
def load(args, prompt_format=None):

    random.seed(42)
    
    print("Loading data...")
    
    corpus = json.load(open(os.path.join(args.data_dir, "MATH.json")))    
    for idx, example in enumerate(corpus):
        example["idx"] = str(idx)
        example["prompt"] = prompt_format.format(input=example["question"]) if prompt_format is not None else example["question"]
        example["answer_str"] = example["answer"][0]
        example["answer_num"] = example["answer"][1]
        del example["answer"]
        
    return Dataset.from_list(corpus)

def eval(args, results):

    corrects = []
    for example in results:
        prediction = utils.answer_clean(args.benchmark, ('####', 'The answer is'), example["prediction"])
        corrects.append(utils.evaluation(args.benchmark, prediction, (example["answer_str"], example["answer_num"])))
        example["predict_answer"] = prediction
    
    metrics = {
        "num": len(results),
        "accuracy": sum(corrects) / len(corrects)
    }
    
    return metrics, results

MATH_EXAMPLARS = [
    {
        "question": "Find the domain of the expression $\\frac{{\sqrt{{x-2}}}}{{\sqrt{{5-x}}}}$.}}",
        "cot_answer": "The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.\nThe answer is $[2,5)$.",
        "short_answer": "$[2,5)$"
    },
    {
        "question": "If $\det \mathbf{{A}} = 2$ and $\det \mathbf{{B}} = 12,$ then find $\det (\mathbf{{A}} \mathbf{{B}}).$",
        "cot_answer": "We have that $\det (\mathbf{{A}} \mathbf{{B}}) = (\det \mathbf{{A}})(\det \mathbf{{B}}) = (2)(12) = \\boxed{{24}}.$\nThe answer is $24$.",
        "short_answer": "24"
    },
    {
        "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "cot_answer": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{{align*}} 30n&=480\\\\ \Rightarrow\qquad n&=480/30=\\boxed{{16}} \end{{align*}}\nThe answer is $16$.",
        "short_answer": "16"
    },
    {
        "question": "If the system of equations: \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.",
        "cot_answer": "If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain $$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{{3}}{{2}}a=b\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$\nThe answer is $-\\frac{{2}}{{3}}$.",
        "short_answer": "$-\\frac{{2}}{{3}}$"
    },
]