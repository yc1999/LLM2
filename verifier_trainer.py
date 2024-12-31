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
import inspect
import warnings
import pyarrow as pa
from dataclasses import FrozenInstanceError, replace, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import Dataset
from accelerate import PartialState
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction

from trl.import_utils import is_peft_available
from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    
from configs import DataArguments


@dataclass
class VerifierConfig(TrainingArguments):
    """
    RewardConfig collects all training arguments related to the [`RewardTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        gradient_checkpointing (`bool`, *optional*, defaults to `True`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """

    max_length: Optional[int] = None
    ddp_find_unused_parameters: bool = True
    
    """The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."""

@dataclass
class VerifierDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = self.tokenizer.pad(
            [{"input_ids": example["input_ids"]} for example in features],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )["input_ids"]

        position_ids = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x["position_ids"]) for x in features], batch_first=True, padding_value=0)
        
        bsz, seqlen = input_ids.shape

        attention_mask = torch.ones(bsz, seqlen, seqlen, dtype=torch.int64)
        for i, x in enumerate(features):
            for j, mask in enumerate(x["attention_mask"]):
                attention_mask[i, j, : len(mask)] = torch.LongTensor([int(x) for x in mask])

        max_pair_num = max(1, max([len(x["pair_idx"]) for x in features]))
        pair_idx = torch.zeros(bsz, max_pair_num, 2, 1).long()
        for i, x in enumerate(features):
            for j, pair in enumerate(x["pair_idx"]):
                pair_idx[i, j] = torch.LongTensor(pair)
         
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pair_idx": pair_idx
        }          
            

class VerifierTrainer(Trainer):

    _tag_names = ["verifier-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[VerifierConfig] = None,
        data_args: Optional[DataArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            max_length (`int`, defaults to `None`):
                The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`."
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                model = get_peft_model(model, peft_config)

        if compute_metrics is None:
            compute_metrics = compute_accuracy
        
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        data_collator = VerifierDataCollatorWithPadding(tokenizer, max_length=args.max_length)

        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.process_verifier_feature, 
                                              num_proc=data_args.preprocessing_num_workers,
                                              keep_in_memory=data_args.keep_in_memory,
                                              load_from_cache_file=data_args.load_from_cache_file,
                                              desc="Processing verifier train dataset")
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.process_verifier_feature, 
                                                num_proc=data_args.preprocessing_num_workers,
                                                keep_in_memory=data_args.keep_in_memory,
                                                load_from_cache_file=data_args.load_from_cache_file,
                                                desc="Processing verifier eval dataset")
 
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        logits = model(**inputs, return_dict=True).logits

        pair_idx = inputs["pair_idx"]
        # [bsz, slen], [bsz, psz, 2, plen] --> [bsz, psz, 2, plen]
        pair_logits = logits.gather(1, pair_idx.view(pair_idx.shape[0], -1)).view(pair_idx.shape)
        # take the mean of the logits for each pair
        pooled_logits = pair_logits.mean(dim=-1)
        
        # [bsz, psz] * [bsz, psz]
        pair_labels = torch.ne(pair_idx.sum(-1).sum(-1), 0).byte().to(pooled_logits.device)
        loss = -(nn.functional.logsigmoid(pooled_logits[:, :, 0] - pooled_logits[:, :, 1]) * pair_labels.float()).mean()
    
        if return_outputs:
            return loss, pooled_logits
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = logits.detach().view(-1, 2).softmax(-1).T
        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)     

        return loss, logits, labels

    def process_verifier_feature(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        
        max_length = self.max_length
        
        input_ids = self.tokenizer.apply_chat_template(feature["messages"], add_generation_prompt=False, tokenize=True)
        seqlen = len(input_ids)
        position_ids = list(range(seqlen))
        attention_mask = [[] for _ in range(seqlen)]
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            position_ids = position_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        pair_idx = []
        for (pos, gt, sample) in feature["samples"]:   
            if pos >= max_length:
                continue
            assert input_ids[pos] == gt
            pair_idx.append([[pos], [len(input_ids)]])
            attention_mask.extend(
                [[1] * pos + [0] * (len(input_ids) - pos) + [1] * 1] * 1
            )
            input_ids = input_ids + [sample]
            position_ids = position_ids + [pos]
        
        assert len(input_ids) == len(attention_mask) == len(position_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": [pa.array([bool(x) for x in mask], type=pa.bool_()) for mask in attention_mask],
            "position_ids": position_ids,
            "pair_idx": pair_idx, 
        }