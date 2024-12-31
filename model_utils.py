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
from pathlib import Path
from typing import Dict, Union

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig, PeftConfig

from trl import ModelConfig
from configs import DataArguments, DPOConfig, SFTConfig


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Union[Dict[str, int], None]:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_quantization_config(model_args: ModelConfig) -> Union[BitsAndBytesConfig, None]:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_tokenizer(
    model_args: ModelConfig, data_args: DataArguments, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    if data_args.chat_template is not None:
        print("Setting chat template to user defined")
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is not None:
        print("Using chat template from tokenizer")
    else:
        print("Setting chat template to default")
        tokenizer.chat_template = tokenizer.default_chat_template
    
    if data_args.pad_token is not None:
        print("Setting pad token to user defined")
        tokenizer.add_special_tokens({"pad_token": data_args.pad_token})
    elif tokenizer.pad_token_id is not None:
        print("Using pad token from tokenizer") 
    else:
        print("Setting pad token to eos token")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side
    
    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048
    
    return tokenizer


def get_peft_config(model_args: ModelConfig) -> Union[PeftConfig, None]:
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


def get_checkpoint(training_args: Union[SFTConfig,DPOConfig]) -> Union[Path,None]:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint
