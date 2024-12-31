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
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    auto_insert_empty_system_msg: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={"help": "The padding token to use for the tokenizer."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The column name to use for the text in the dataset (only used for continued pretraining)."},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_in_memory: bool = field(
        default=True,
        metadata={"help": "Whether to keep the dataset in memory or not."},
    )
    load_from_cache_file: bool = field(
        default=False,
        metadata={"help": "Whether to load the dataset from the cache file or not."},
    )
    load_from_disk: bool = field(
        default=False,
        metadata={"help": "Whether to load the dataset from disk or not."},
    )

@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """

    dataset_kwargs: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Dataset kwargs for the SFTTrainer"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    packing: Optional[str] = field(
        default=False,
        metadata={"help": "Whether to pack the sequences or not."},
    )

@dataclass
class DPOConfig(transformers.TrainingArguments):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": ("For DPO, the maximum length of the prompt to use for conditioning the model.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": ("The loss type for DPO.")})


@dataclass
class ORPOConfig(transformers.TrainingArguments):
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the sequences in the batch."},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the prompt."},
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the completions."},
    )

    beta: float = field(
        default=0.1,
        metadata={
            "help": "The beta factor in ORPO loss (lambda/alpha in paper/code) that is the weight of the relative loss ratio in the SFT loss."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether or not to disable dropouts in `model`."},
    )

    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "The label pad token id."},
    )
    padding_value: Optional[int] = field(
        default=None,
        metadata={"help": "The padding value if it is different to the tokenizer's pad_token_id."},
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={"help": "The truncation mode to use, either `keep_end` or `keep_start`."},
    )

    generate_during_eval: bool = field(
        default=False,
        metadata={"help": "Whether to sample and log generations during evaluation step."},
    )
    is_encoder_decoder: Optional[bool] = field(
        default=None,
        metadata={"help": ("If no model is provided, we need to know if the model_init returns an encoder-decoder.")},
    )

    model_init_kwargs: Optional[Dict] = field(
        default=None,
        metadata={"help": ("Dict of Optional kwargs to pass when instantiating the model from a string")},
    )

    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": ("The number of workers to use to tokenize the data.")},
    )
