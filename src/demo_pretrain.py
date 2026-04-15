import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data_setup import collate_fn, setup_dataloaders


def build_pretrain_cfg(yaml_cfg):
    return {
        "model_name": yaml_cfg["model"]["name"],
        "model": {
            "student": {
                "reduce_size": yaml_cfg["model"].get("student", {}).get("reduce_size", True),
                "size_reduction_factor": yaml_cfg["model"].get("student", {}).get("size_reduction_factor", 2),
            }
        },
        "data_path": yaml_cfg["data"]["sources"][0]["path"],
        "max_length": yaml_cfg["data"]["max_length"],
        "stride": yaml_cfg["data"]["stride"],
        "batch_size": yaml_cfg["data"]["batch_size"],
        "seed": yaml_cfg["training"]["seed"],
        "val_split_ratio": yaml_cfg["data"].get("val_split_ratio", 0.1),
        "val_stride": yaml_cfg["data"].get("val_stride"),
        "train_padding_side": yaml_cfg.get("tokenizer", {}).get("train_padding_side", "right"),
        "generation_padding_side": yaml_cfg.get("tokenizer", {}).get("generation_padding_side", "left"),
        "generation": yaml_cfg.get("generation", {}),
    }


class CachedTokenizedDataset(Dataset):
    def __init__(self, cache_obj):
        self.cache_obj = cache_obj
        self.use_tensor_cache = isinstance(cache_obj, dict) and "input_ids" in cache_obj and "attention_mask" in cache_obj

    def __len__(self):
        if self.use_tensor_cache:
            return self.cache_obj["input_ids"].size(0)
        return len(self.cache_obj)

    def __getitem__(self, idx):
        if self.use_tensor_cache:
            input_ids = self.cache_obj["input_ids"][idx].to(torch.long)
            attention_mask = self.cache_obj["attention_mask"][idx].to(torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

        item = self.cache_obj[idx]
        return {
            "input_ids": item["input_ids"].clone().detach(),
            "attention_mask": item["attention_mask"].clone().detach(),
            "labels": item["labels"].clone().detach(),
        }


def load_student_model(cfg, checkpoint_path, device):
    model_config = AutoConfig.from_pretrained(cfg["model_name"])
    if cfg["model"]["student"].get("reduce_size", False):
        factor = cfg["model"]["student"].get("size_reduction_factor", 1)
        model_config.num_hidden_layers = max(1, model_config.num_hidden_layers // factor)
        model_config.intermediate_size = max(1, model_config.intermediate_size // factor)

    model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["student_model_state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys while loading student checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys while loading student checkpoint: {unexpected_keys}")
    model.eval()
    return model


def load_teacher_model(cfg, device):
    teacher_dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.startswith("cuda") else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=teacher_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def load_demo_loader(cfg, tokenizer, split, cache_pt):
    if cache_pt:
        cache_obj = torch.load(cache_pt, map_location="cpu")
        dataset = CachedTokenizedDataset(cache_obj)
        return DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    train_loader, val_loader = setup_dataloaders(cfg, tokenizer)
    return train_loader if split == "train" else val_loader


def build_generate_kwargs(tokenizer, generation_cfg, max_new_tokens_override):
    do_sample = generation_cfg.get("do_sample", False)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens_override or generation_cfg.get("max_new_tokens", 50),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = generation_cfg.get("temperature", 0.7)
        generate_kwargs["top_p"] = generation_cfg.get("top_p", 0.9)
    else:
        generate_kwargs["temperature"] = 1.0
        generate_kwargs["top_p"] = 1.0
    return generate_kwargs


def generate_samples(
    student_model,
    teacher_model,
    tokenizer,
    dataloader,
    device,
    num_samples,
    generation_cfg,
    max_new_tokens_override,
    prompt_max_tokens,
):
    samples = []
    generate_kwargs = build_generate_kwargs(tokenizer, generation_cfg, max_new_tokens_override)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = generation_cfg.get("padding_side", original_padding_side)

    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                for i in range(input_ids.size(0)):
                    if len(samples) >= num_samples:
                        return samples

                    sequence_length = int(attention_mask[i].sum().item())
                    if sequence_length <= 1:
                        continue

                    # Match pretraining sample logging: prompt is the first half.
                    prompt_length = max(1, min(sequence_length - 1, sequence_length // 2))
                    prompt_ids = input_ids[i][:prompt_length]
                    ground_truth_ids = input_ids[i][prompt_length:sequence_length]

                    # Keep only the tail of the prompt for cleaner qualitative demos.
                    if prompt_max_tokens is not None:
                        prompt_ids = prompt_ids[-prompt_max_tokens:]

                    generation_input_ids = prompt_ids.unsqueeze(0)
                    generation_attention_mask = torch.ones_like(generation_input_ids)

                    student_output = student_model.generate(
                        input_ids=generation_input_ids,
                        attention_mask=generation_attention_mask,
                        **generate_kwargs,
                    )
                    student_completion = tokenizer.decode(
                        student_output[0][generation_input_ids.shape[1]:],
                        skip_special_tokens=True,
                    )

                    teacher_completion = None
                    if teacher_model is not None:
                        teacher_output = teacher_model.generate(
                            input_ids=generation_input_ids,
                            attention_mask=generation_attention_mask,
                            **generate_kwargs,
                        )
                        teacher_completion = tokenizer.decode(
                            teacher_output[0][generation_input_ids.shape[1]:],
                            skip_special_tokens=True,
                        )

                    samples.append(
                        {
                            "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                            "ground_truth": tokenizer.decode(ground_truth_ids, skip_special_tokens=True),
                            "student_completion": student_completion,
                            "teacher_completion": teacher_completion,
                            "prompt_token_count": int(prompt_ids.numel()),
                            "ground_truth_token_count": int(ground_truth_ids.numel()),
                        }
                    )
        return samples
    finally:
        tokenizer.padding_side = original_padding_side


def print_samples(samples, include_teacher):
    for idx, sample in enumerate(samples, start=1):
        print(f"\n=== Sample {idx} ===")
        print(f"Prompt:\n{sample['prompt']}\n")
        print(f"Ground truth:\n{sample['ground_truth']}\n")
        print(f"Student completion:\n{sample['student_completion']}\n")
        if include_teacher:
            print(f"Teacher completion:\n{sample['teacher_completion']}\n")


def maybe_save_report(output_path, args, cfg, samples):
    if not output_path:
        return

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "cache_pt": args.cache_pt,
        "split": args.split,
        "num_samples": args.num_samples,
        "prompt_max_tokens": args.prompt_max_tokens,
        "max_new_tokens": args.max_new_tokens or cfg["generation"].get("max_new_tokens", 50),
        "include_teacher": args.include_teacher,
        "model_name": cfg["model_name"],
        "samples": samples,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate short pretraining-style continuation demos from a trained student checkpoint."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to pretraining config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to student checkpoint")
    parser.add_argument("--cache_pt", type=str, default=None, help="Optional tokenized dataset cache .pt file to read directly")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val", help="Dataset split to sample from when --cache_pt is not provided")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to print")
    parser.add_argument("--prompt_max_tokens", type=int, default=128, help="Use only the last N prompt tokens for generation and display")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override generation max_new_tokens")
    parser.add_argument("--include_teacher", action="store_true", help="Also generate teacher continuations for side-by-side comparison")
    parser.add_argument("--output_path", type=str, default=None, help="Optional JSON path for saving the printed samples")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    cfg = build_pretrain_cfg(yaml_cfg)
    cfg["generation"]["padding_side"] = cfg["generation_padding_side"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg["train_padding_side"]

    dataloader = load_demo_loader(cfg, tokenizer, args.split, args.cache_pt)
    student_model = load_student_model(cfg, args.checkpoint, device)
    teacher_model = load_teacher_model(cfg, device) if args.include_teacher else None

    samples = generate_samples(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        num_samples=args.num_samples,
        generation_cfg=cfg["generation"],
        max_new_tokens_override=args.max_new_tokens,
        prompt_max_tokens=args.prompt_max_tokens,
    )

    print_samples(samples, args.include_teacher)
    maybe_save_report(args.output_path, args, cfg, samples)


if __name__ == "__main__":
    main()
