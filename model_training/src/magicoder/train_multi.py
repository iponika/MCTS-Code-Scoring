from dataclasses import dataclass, field
from typing import cast
import inspect
import gc
import importlib.util
import os
import json
import torch
import shutil
import numpy as np
from datasets import load_dataset, DatasetDict
#from magicoder.functions import *
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import HfArgumentParser, Trainer, TrainingArguments, TrainerCallback, PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
)

from typing import TYPE_CHECKING, Any, Dict, Optional

from magicoder.llm_wrapper import (
    DecodingConfig,
    EncodingConfig,
    TokenizationContext,
    get_model_context,
    get_model_wvalue_context,
    pad_sequences,
)
from magicoder.prompt_template import DSC_PROMPT, QWEN_REVIEW_STEP_PROMPT, QWEN_STEP_PROMPT
from magicoder.utils import N_CORES
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn.functional as F


if importlib.util.find_spec("safetensors") is not None:
    from safetensors import safe_open
    from safetensors.torch import save_file

V_HEAD_WEIGHTS_NAME = "value_head.pth"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"

model_name = ''
value_weight = ''



@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None
    peft: str | None = None



@dataclass(frozen=True)
class Args:
    datafile_paths: list[str] = field(default_factory=list)
    max_training_seq_length: int = field(default=1216)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )
    use_flash_attention: bool = field(default=False)
    value_weight: float = field(default=0.2)
    task: str = field(default="code", metadata={"help": "code or review"})
    skip_save: bool = field(default=False, metadata={"help": "Skip final model saving for smoke tests."})
    num_proc: int = field(default=20, metadata={"help": "Number of dataset preprocessing workers."})


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100


def map_dataset(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    model_inputs = {"input_ids": [], "attention_mask": [], "Q": [], "labels": [], "exceeding_length":[]}

    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        responses = examples["response"][i]
        q_value = examples["q_value"][i]
        train_lm_flags = examples.get("train_lm")
        train_lm = train_lm_flags[i] if train_lm_flags is not None else None

 
        if args.task == "review":
            prompt = QWEN_REVIEW_STEP_PROMPT.format(instruction=instruction, response="")
        elif 'deepseek' in model_name or 'dsc' in model_name:
            prompt = DSC_PROMPT.format(instruction=instruction, response="<step>\n")
        else:
            prompt = QWEN_STEP_PROMPT.format(instruction=instruction, response="<step>\n")
        prompt_config = EncodingConfig(add_bos=True, add_eos=False)
        prompt_id_batches = context.encode(prompt_config, [prompt])
        input_ids = prompt_id_batches[0]

        
        Q = [IGNORED_INDEX] * len(input_ids)
        labels = [IGNORED_INDEX] * len(input_ids)
        
        response_state = q_value[-1]
        for response,q in zip(responses, q_value):
            sub_Q = float(q)
            completion_config = EncodingConfig(add_bos=False, add_eos=False)
            completion_id_batches = context.encode(completion_config, [response.strip()+'\n'])
            sub_response_ids = completion_id_batches[0]

            input_ids += sub_response_ids
            if sub_Q == IGNORED_INDEX or (args.task != "review" and all(x == 1 for x in q_value) and len(q_value) > 1):
                Q += [IGNORED_INDEX] * (len(sub_response_ids))
            else:
                Q += [IGNORED_INDEX] * (len(sub_response_ids) - 1) + [sub_Q]
            labels += sub_response_ids

            if len(input_ids) > args.max_training_seq_length:
                break

        input_ids += [context.tokenizer.eos_token_id]
        Q += [IGNORED_INDEX]
        labels += [context.tokenizer.eos_token_id]

        if len(input_ids) > args.max_training_seq_length:
            #continue
            input_ids = input_ids[:args.max_training_seq_length]
            Q = Q[:args.max_training_seq_length]
            labels = labels[:args.max_training_seq_length]
            model_inputs["exceeding_length"].append(True)
        else:
            model_inputs["exceeding_length"].append(False)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["Q"].append(np.array(Q, dtype=np.float32))


        if train_lm is None:
            train_lm = response_state == 1 or all(x == IGNORED_INDEX for x in q_value)
        if train_lm:
            model_inputs["labels"].append(labels)
        else:
            model_inputs["labels"].append([IGNORED_INDEX] * len(labels))

    return model_inputs


def get_data_collator(args: "Args", pad_token_id: int):
    """Pad input_ids to the right, create labels by setting the padding tokens to -100, and
    create attention_mask to ignore the padding tokens"""

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        q_unpadded = [example["Q"] for example in examples]
        padding_length = (
            args.max_training_seq_length if args.pad_to_max_length else None
        )
        input_ids = pad_sequences(
            input_ids_unpadded, pad_token_id, "right", padding_length=padding_length
        )
        q = pad_sequences(
            q_unpadded, IGNORED_INDEX, "right", padding_length=padding_length, dtype=torch.float32
        )
        labels = pad_sequences(
            labels_unpadded, IGNORED_INDEX, "right", padding_length=padding_length
        )

        assert input_ids.shape == labels.shape
        assert len(input_ids) == len(examples)
        # Enforced in `map_raw_dataset`
        assert input_ids.shape[-1] <= args.max_training_seq_length
        if args.pad_to_max_length:
            assert input_ids.shape[-1] == args.max_training_seq_length

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
            "Q": q
        }

    return collate


class RLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0
        self._loss_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        torch.cuda.empty_cache()
        gc.collect()
        # Compute rewards
        Q = inputs.get("Q", None)
        if Q is not None:
            del inputs["Q"]
        
        labels = inputs.get("labels", None)
        if labels is not None:
            del inputs["labels"]
            
        mask = Q.ne(IGNORED_INDEX)

        lm_logits, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        values = torch.tanh(values)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        if torch.all(shift_labels==IGNORED_INDEX):
            loss_fct = CrossEntropyLoss(reduction='sum')
        else:
            loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.pretrained_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        assert not torch.isnan(loss) and Q is not None

        Q = Q.type_as(values)
        masked_values = torch.where(mask, values, Q)
        value_loss = F.mse_loss(masked_values, Q, reduction='sum') / (mask.sum() + 1e-3)
        all_losses =  loss + value_weight * value_loss


        if return_outputs:
            return all_losses, [all_losses, value_loss, value_loss, masked_values, Q]
        return all_losses #, value_loss

        
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        fix_valuehead_checkpoint(
            model=self.model,
            output_dir=output_dir or self.args.output_dir,
            safe_serialization=self.args.save_safetensors
        )





def train():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, Args))
    model_args, training_args, args = cast(
        tuple[ModelArguments, TrainingArguments, Args],
        parser.parse_args_into_dataclasses(),
    )


    training_args.gradient_checkpointing=True
    training_args.remove_unused_columns=False
    training_args.save_safetensors=False
    global model_name
    model_name = model_args.model_key
    global value_weight
    value_weight = args.value_weight
    dataset = load_dataset("json", data_files=args.datafile_paths, split="train")
    
    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key
    
    
    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    dataset_num_proc = args.num_proc if args.num_proc > 1 else None
    
    train_dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args, context=tokenization_context),
        batched=True,
        num_proc=dataset_num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,  # not args.overwrite_cache
        desc="Running tokenizer on train dataset",
    )

    msg = f"#Examples truncated: {sum(train_dataset['exceeding_length'])} / {len(train_dataset)}"
    train_dataset = train_dataset.filter(
        lambda x: not x['exceeding_length'],
        desc="Removing examples exceeding max length",
        num_proc=dataset_num_proc,
    )
    print(f"Dataset size after filtering: {len(train_dataset)}")
    print(msg)


    # Shuffling
    if hasattr(training_args, "eval_strategy"):
        training_args.eval_strategy = "no"
        evaluation_strategy = training_args.eval_strategy
    else:
        training_args.evaluation_strategy = "no"
        evaluation_strategy = training_args.evaluation_strategy
    if training_args.eval_steps is None and evaluation_strategy == "no":
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        eval_dataset = None
    else:
        print("Splitting dataset")
        split_dataset = train_dataset.train_test_split(
            test_size=args.eval_dataset_size,
            shuffle=True,
            seed=training_args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    state = get_model_wvalue_context(
        model_key,
        model_name_or_path,
        tokenization_context,
        inference_mode=False,
        use_flash_attention=args.use_flash_attention,
        model_args=model_args
    )
    state.model.config.use_cache = False

    print("Parallel mode:", training_args.parallel_mode)
    data_collator = get_data_collator(args, state.tokenization_context.pad_token_id)


    trainer = RLTrainer(
        model=state.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if args.skip_save:
        return
    trainer.save_model(training_args.output_dir)
    state.tokenization_context.tokenizer.save_pretrained(training_args.output_dir)

    if model_args.peft=='lora':
        peft_model = state.model.pretrained_model
        merged_model  = peft_model.merge_and_unload()
        merged_model.save_pretrained(training_args.output_dir, safe_serialization=True, max_shard_size="1000GB")
    
        if not os.path.exists(os.path.join(training_args.output_dir, 'peft')):
            os.makedirs(os.path.join(training_args.output_dir, 'peft'))
            shutil.move(os.path.join(training_args.output_dir, 'adapter_config.json'), os.path.join(training_args.output_dir, 'peft'))
            shutil.move(os.path.join(training_args.output_dir, 'adapter_model.safetensors'), os.path.join(training_args.output_dir, 'peft'))

def fix_valuehead_checkpoint(
    model, output_dir, safe_serialization
) -> None:
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    index_file = os.path.join(output_dir, "pytorch_model.bin.index.json")
    if os.path.exists(os.path.join(output_dir, WEIGHTS_NAME)):
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict = torch.load(path_to_checkpoint, map_location="cpu")
        os.remove(path_to_checkpoint)
    elif os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index_data = json.load(f)
            
        state_dict = {}
        print(set(index_data["weight_map"].values()))
        for weight_file in set(index_data["weight_map"].values()):
            path = os.path.join(output_dir, weight_file)
            state_dict.update(torch.load(path, map_location="cpu"))
            os.remove(path)


    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param
    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=True, max_shard_size="1000GB"
    )
    torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))



if __name__ == "__main__":
    train()
