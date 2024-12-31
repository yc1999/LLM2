import os
import re
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import evaluate
from datasets import Dataset

TASK_TYPE = "generation"

GSM_EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8"
    }
]


def load(args, prompt_format=None):

    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
        
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    global GSM_EXAMPLARS
    if args.n_shot > 0:
        if len(GSM_EXAMPLARS) > args.n_shot:
            GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
        demonstrations = []
        for example in GSM_EXAMPLARS:
            if args.no_cot:
                demonstrations.append(
                    "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                )
            else:
                demonstrations.append(
                    "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
                )
        # prompt_prefix = "\n\n".join(demonstrations) + "\n\n" + "Please answer the question based on above examples:\n\n" + "Question: "
        prompt_prefix = "\n\n".join(demonstrations) + "\n\n" + "Question: "
    else:
        prompt_prefix = ""

    for idx, example in enumerate(test_data):
        example["idx"] = str(idx)
        if args.n_shot > 0:
            prompt = prompt_prefix + example["question"].strip() + "\n" + "Answer: "
        else:
            prompt = prompt_prefix + example["question"].strip()
        example["prompt"] = prompt_format.format(input=prompt) if prompt_format is not None else prompt
        
    return Dataset.from_list(test_data)

def eval(args, results):

    for example in results:
        # replace numbers like `x,xxx` with `xxxx`
        prediction = re.sub(r"(\d),(\d)", r"\1\2", example["prediction"])
        predict_answer = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)

        example["predict_answer"] = predict_answer[-1] if predict_answer else prediction

    scores = [int(x == y) for x, y in zip([example["predict_answer"] for example in results], [example["answer"] for example in results])]
    
    metrics = {
        "num": len(results),
        "accuracy": np.mean(scores),
    }
    
    return metrics, results