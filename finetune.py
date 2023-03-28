import os
from dataclasses import dataclass, field

import datasets
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer
from transformers import Trainer, HfArgumentParser
from transformers import TrainingArguments

from modeling_chatglm import ChatGLMForConditionalGeneration

@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="THUDM/chatglm-6b")
    lora_rank: int = field(default=8)
    is_resume: bool = field(default=False)
    resume_path: str = field(default='output/alpaca_output', )


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_masks_and_position_ids(
        seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
            seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position
    return attention_mask, position_ids


def data_collator(features: list, eos_token_id: int) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [-100] * (seq_len - 1)
                + ids[(seq_len - 1):]
                + [eos_token_id]
                + [-100] * (longest - ids_l - 1)
        )
        ids = ids + [eos_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)


def main():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)
    # init model
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = ChatGLMForConditionalGeneration.from_pretrained(
        finetune_args.model_path, load_in_8bit=False, trust_remote_code=True, device_map=device_map
    )

    # model = ChatGLMForConditionalGeneration.from_pretrained(
    #     "../../pretrained_models/chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map=device_map
    # )

    # model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    if finetune_args.is_resume and finetune_args.resume_path:
        print("=====>load lora pt from =====》:", finetune_args.is_resume, finetune_args.resume_path)
        model.load_state_dict(torch.load(finetune_args.resume_path), strict=False)

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=lambda b: data_collator(b, tokenizer.eos_token_id),
    )
    trainer.train()

    # save model
    save_tunable_parameters(
        model, os.path.join(training_args.output_dir, "chatglm-lora.pt")
    )


if __name__ == "__main__":
    main()
