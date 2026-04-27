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
    from safetensors.torch import load_file, save_file

V_HEAD_WEIGHTS_NAME = "value_head.pth"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"

model_name = ''
value_weight = ''
boundary_value_weight = 0.0
pairwise_value_weight = 0.0
pairwise_margin = 0.2


def optional_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_q_sequence(value: Any, fallback: list[Any]) -> list[float]:
    if value is None:
        value = fallback
    if not isinstance(value, list):
        value = [value] * len(fallback)
    normalized: list[float] = []
    for idx, fallback_value in enumerate(fallback):
        item = value[idx] if idx < len(value) else fallback_value
        normalized.append(optional_float(item, optional_float(fallback_value, float(IGNORED_INDEX))))
    return normalized



@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None
    peft: str | None = None
    lora_rank: int = 64
    lora_alpha: int | None = None
    lora_dropout: float = 0.1
    lora_target_scope: str = field(
        default="all",
        metadata={"help": "LoRA target scope: all, attention, or attention_mlp. Defaults to the previous full target list."},
    )



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
    boundary_value_weight: float = field(
        default=0.0,
        metadata={"help": "Optional auxiliary margin loss for the AXIOM functional boundary; 0 disables it."},
    )
    pairwise_value_weight: float = field(
        default=0.0,
        metadata={"help": "Optional margin-ranking loss for paired value labels; 0 disables it."},
    )
    pairwise_margin: float = field(default=0.2, metadata={"help": "Required value margin between positive and negative paired samples."})
    disable_train_shuffle: bool = field(default=False, metadata={"help": "Keep tokenized training examples in dataset order."})
    task: str = field(default="code", metadata={"help": "code or review"})
    skip_save: bool = field(default=False, metadata={"help": "Skip final model saving for smoke tests."})
    force_gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable Trainer gradient checkpointing. Disable when using FSDP activation checkpointing."})
    save_merged_model: bool = field(
        default=False,
        metadata={"help": "For LoRA runs, also save a merged full-backbone checkpoint. Disabled by default to keep smoke checkpoints small."},
    )
    num_proc: int = field(default=20, metadata={"help": "Number of dataset preprocessing workers."})


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100


def map_dataset(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "Q": [],
        "Q_MIN": [],
        "Q_MAX": [],
        "labels": [],
        "value_loss_weight": [],
        "lm_loss_weight": [],
        "pair_id": [],
        "pair_role": [],
        "exceeding_length": [],
    }

    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        responses = examples["response"][i]
        q_value = examples["q_value"][i]
        q_min = normalize_q_sequence(examples.get("q_min", [None] * len(examples["instruction"]))[i] if examples.get("q_min") is not None else None, q_value)
        q_max = normalize_q_sequence(examples.get("q_max", [None] * len(examples["instruction"]))[i] if examples.get("q_max") is not None else None, q_value)
        train_lm_flags = examples.get("train_lm")
        train_lm = train_lm_flags[i] if train_lm_flags is not None else None
        value_loss_weights = examples.get("value_loss_weight")
        lm_loss_weights = examples.get("lm_loss_weight")
        value_loss_weight = optional_float(value_loss_weights[i], 1.0) if value_loss_weights is not None else 1.0
        lm_loss_weight = optional_float(lm_loss_weights[i], 1.0) if lm_loss_weights is not None else 1.0
        pair_ids = examples.get("pair_id")
        pair_roles = examples.get("pair_role")
        pair_id = "" if pair_ids is None or pair_ids[i] is None else str(pair_ids[i])
        pair_role = "" if pair_roles is None or pair_roles[i] is None else str(pair_roles[i])

 
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
        Q_MIN = [IGNORED_INDEX] * len(input_ids)
        Q_MAX = [IGNORED_INDEX] * len(input_ids)
        labels = [IGNORED_INDEX] * len(input_ids)
        
        response_state = q_value[-1]
        for response, q, q_lower, q_upper in zip(responses, q_value, q_min, q_max):
            sub_Q = float(q)
            sub_Q_MIN = float(q_lower)
            sub_Q_MAX = float(q_upper)
            completion_config = EncodingConfig(add_bos=False, add_eos=False)
            completion_id_batches = context.encode(completion_config, [response.strip()+'\n'])
            sub_response_ids = completion_id_batches[0]

            input_ids += sub_response_ids
            if sub_Q == IGNORED_INDEX or (args.task != "review" and all(x == 1 for x in q_value) and len(q_value) > 1):
                Q += [IGNORED_INDEX] * (len(sub_response_ids))
                Q_MIN += [IGNORED_INDEX] * (len(sub_response_ids))
                Q_MAX += [IGNORED_INDEX] * (len(sub_response_ids))
            else:
                Q += [IGNORED_INDEX] * (len(sub_response_ids) - 1) + [sub_Q]
                Q_MIN += [IGNORED_INDEX] * (len(sub_response_ids) - 1) + [sub_Q_MIN]
                Q_MAX += [IGNORED_INDEX] * (len(sub_response_ids) - 1) + [sub_Q_MAX]
            labels += sub_response_ids

            if len(input_ids) > args.max_training_seq_length:
                break

        input_ids += [context.tokenizer.eos_token_id]
        Q += [IGNORED_INDEX]
        Q_MIN += [IGNORED_INDEX]
        Q_MAX += [IGNORED_INDEX]
        labels += [context.tokenizer.eos_token_id]

        if len(input_ids) > args.max_training_seq_length:
            #continue
            input_ids = input_ids[:args.max_training_seq_length]
            Q = Q[:args.max_training_seq_length]
            Q_MIN = Q_MIN[:args.max_training_seq_length]
            Q_MAX = Q_MAX[:args.max_training_seq_length]
            labels = labels[:args.max_training_seq_length]
            model_inputs["exceeding_length"].append(True)
        else:
            model_inputs["exceeding_length"].append(False)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["Q"].append(np.array(Q, dtype=np.float32))
        model_inputs["Q_MIN"].append(np.array(Q_MIN, dtype=np.float32))
        model_inputs["Q_MAX"].append(np.array(Q_MAX, dtype=np.float32))


        if train_lm is None:
            train_lm = response_state == 1 or all(x == IGNORED_INDEX for x in q_value)
        if train_lm:
            model_inputs["labels"].append(labels)
        else:
            model_inputs["labels"].append([IGNORED_INDEX] * len(labels))
        model_inputs["value_loss_weight"].append(value_loss_weight)
        model_inputs["lm_loss_weight"].append(lm_loss_weight)
        model_inputs["pair_id"].append(pair_id)
        model_inputs["pair_role"].append(pair_role)

    return model_inputs


def get_data_collator(args: "Args", pad_token_id: int):
    """Pad input_ids to the right, create labels by setting the padding tokens to -100, and
    create attention_mask to ignore the padding tokens"""

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        q_unpadded = [example["Q"] for example in examples]
        q_min_unpadded = [example.get("Q_MIN", example["Q"]) for example in examples]
        q_max_unpadded = [example.get("Q_MAX", example["Q"]) for example in examples]
        value_loss_weight = torch.tensor([example.get("value_loss_weight", 1.0) for example in examples], dtype=torch.float32)
        lm_loss_weight = torch.tensor([example.get("lm_loss_weight", 1.0) for example in examples], dtype=torch.float32)
        pair_ids = [str(example.get("pair_id", "") or "") for example in examples]
        pair_roles = [str(example.get("pair_role", "") or "") for example in examples]
        padding_length = (
            args.max_training_seq_length if args.pad_to_max_length else None
        )
        input_ids = pad_sequences(
            input_ids_unpadded, pad_token_id, "right", padding_length=padding_length
        )
        q = pad_sequences(
            q_unpadded, IGNORED_INDEX, "right", padding_length=padding_length, dtype=torch.float32
        )
        q_min = pad_sequences(
            q_min_unpadded, IGNORED_INDEX, "right", padding_length=padding_length, dtype=torch.float32
        )
        q_max = pad_sequences(
            q_max_unpadded, IGNORED_INDEX, "right", padding_length=padding_length, dtype=torch.float32
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
            "Q": q,
            "Q_MIN": q_min,
            "Q_MAX": q_max,
            "value_loss_weight": value_loss_weight,
            "lm_loss_weight": lm_loss_weight,
            "pair_id": pair_ids,
            "pair_role": pair_roles,
        }

    return collate


def normalize_pair_role(role: str) -> str:
    role = str(role or "").lower()
    if role in {"pos", "positive", "pos_response"}:
        return "pos"
    if role in {"neg", "negative", "neg_response"}:
        return "neg"
    return ""


def pairwise_margin_loss(
    values: torch.Tensor,
    mask: torch.Tensor,
    pair_ids: list[str] | None,
    pair_roles: list[str] | None,
) -> torch.Tensor:
    if not pair_ids or not pair_roles or pairwise_value_weight <= 0:
        return values.new_tensor(0.0)
    terminal_values: list[torch.Tensor | None] = []
    for sample_idx in range(values.shape[0]):
        positions = torch.nonzero(mask[sample_idx], as_tuple=False).flatten()
        terminal_values.append(values[sample_idx, positions[-1]] if positions.numel() > 0 else None)

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for pair_id, role, value in zip(pair_ids, pair_roles, terminal_values):
        if value is None or not pair_id:
            continue
        normalized_role = normalize_pair_role(role)
        if normalized_role not in {"pos", "neg"}:
            continue
        grouped.setdefault(str(pair_id), {})[normalized_role] = value

    losses = []
    for pair in grouped.values():
        if "pos" not in pair or "neg" not in pair:
            continue
        losses.append(F.relu(values.new_tensor(pairwise_margin) - (pair["pos"] - pair["neg"])) ** 2)
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()


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
        Q_MIN = inputs.get("Q_MIN", None)
        if Q_MIN is not None:
            del inputs["Q_MIN"]
        Q_MAX = inputs.get("Q_MAX", None)
        if Q_MAX is not None:
            del inputs["Q_MAX"]
        value_loss_weight_tensor = inputs.get("value_loss_weight", None)
        if value_loss_weight_tensor is not None:
            del inputs["value_loss_weight"]
        lm_loss_weight_tensor = inputs.get("lm_loss_weight", None)
        if lm_loss_weight_tensor is not None:
            del inputs["lm_loss_weight"]
        pair_ids = inputs.get("pair_id")
        if pair_ids is not None:
            del inputs["pair_id"]
        pair_roles = inputs.get("pair_role")
        if pair_roles is not None:
            del inputs["pair_role"]
        
        labels = inputs.get("labels", None)
        if labels is not None:
            del inputs["labels"]
            
        mask = Q.ne(IGNORED_INDEX)
        has_lm_labels = labels is not None and labels.ne(IGNORED_INDEX).any()
        model_inputs = dict(inputs)
        if not has_lm_labels:
            # Qwen-style CausalLM heads can avoid materializing full
            # sequence-by-vocabulary logits. Value-only batches still need all
            # hidden states for the value head, but not the LM logits.
            model_inputs["logits_to_keep"] = 1

        lm_logits, _, values = model(**model_inputs, output_hidden_states=True, return_dict=True)
        values = torch.tanh(values)

        if value_loss_weight_tensor is None:
            value_loss_weight_tensor = torch.ones(Q.shape[0], device=Q.device, dtype=torch.float32)
        else:
            value_loss_weight_tensor = value_loss_weight_tensor.to(Q.device, dtype=torch.float32)
        if lm_loss_weight_tensor is None:
            lm_loss_weight_tensor = torch.ones(Q.shape[0], device=Q.device, dtype=torch.float32)
        else:
            lm_loss_weight_tensor = lm_loss_weight_tensor.to(Q.device, dtype=torch.float32)

        if has_lm_labels:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            token_loss = F.cross_entropy(
                shift_logits.view(-1, model.pretrained_model.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=IGNORED_INDEX,
                reduction="none",
            ).view_as(shift_labels)
            lm_mask = shift_labels.ne(IGNORED_INDEX)
            per_sample_lm_loss = token_loss.sum(dim=1) / (lm_mask.sum(dim=1).float() + 1e-6)
            effective_lm_weight = lm_loss_weight_tensor.to(shift_logits.device) * lm_mask.any(dim=1).float()
            loss = (per_sample_lm_loss * effective_lm_weight).sum() / (effective_lm_weight.sum() + 1e-6)
        else:
            loss = values.new_tensor(0.0)

        assert not torch.isnan(loss) and Q is not None

        Q = Q.type_as(values)
        if Q_MIN is None:
            Q_MIN = Q
        if Q_MAX is None:
            Q_MAX = Q
        Q_MIN = Q_MIN.type_as(values).to(values.device)
        Q_MAX = Q_MAX.type_as(values).to(values.device)
        mask = mask.to(values.device)
        Q = Q.to(values.device)
        value_loss_weight_tensor = value_loss_weight_tensor.to(values.device)
        lower = torch.minimum(Q_MIN, Q_MAX)
        upper = torch.maximum(Q_MIN, Q_MAX)
        interval_error = F.relu(lower - values) ** 2 + F.relu(values - upper) ** 2
        value_error = torch.where(mask, interval_error, torch.zeros_like(values))
        per_sample_value_loss = value_error.sum(dim=1) / (mask.sum(dim=1).float() + 1e-6)
        effective_value_weight = value_loss_weight_tensor * mask.any(dim=1).float()
        value_loss = (per_sample_value_loss * effective_value_weight).sum() / (effective_value_weight.sum() + 1e-6)
        target_sign = torch.where(Q >= 0, torch.ones_like(Q), -torch.ones_like(Q))
        boundary_error = F.relu(0.1 - target_sign * values) ** 2
        boundary_error = torch.where(mask, boundary_error, torch.zeros_like(values))
        per_sample_boundary_loss = boundary_error.sum(dim=1) / (mask.sum(dim=1).float() + 1e-6)
        boundary_loss = (per_sample_boundary_loss * effective_value_weight).sum() / (effective_value_weight.sum() + 1e-6)
        pairwise_loss = pairwise_margin_loss(values, mask, pair_ids, pair_roles)
        masked_values = torch.where(mask, values, Q)
        all_losses =  loss + value_weight * value_loss + boundary_value_weight * boundary_loss + pairwise_value_weight * pairwise_loss


        if return_outputs:
            return all_losses, [all_losses, value_loss, boundary_loss, pairwise_loss, masked_values, Q]
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

    training_args.gradient_checkpointing=args.force_gradient_checkpointing
    training_args.remove_unused_columns=False
    training_args.save_safetensors=False
    global model_name
    model_name = model_args.model_key
    global value_weight, boundary_value_weight, pairwise_value_weight, pairwise_margin
    value_weight = args.value_weight
    boundary_value_weight = args.boundary_value_weight
    pairwise_value_weight = args.pairwise_value_weight
    pairwise_margin = args.pairwise_margin
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
        if not args.disable_train_shuffle:
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

    if model_args.peft=='lora' and args.save_merged_model:
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

    state_dict = None
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
    elif importlib.util.find_spec("safetensors") is not None and os.path.exists(os.path.join(output_dir, SAFE_WEIGHTS_NAME)):
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        state_dict = load_file(path_to_checkpoint, device="cpu")
        os.remove(path_to_checkpoint)

    decoder_state_dict, v_head_state_dict = {}, {}
    if state_dict is None:
        v_head_state_dict = {
            f"v_head.{name}": param.detach().cpu()
            for name, param in model.v_head.state_dict().items()
        }
    else:
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
