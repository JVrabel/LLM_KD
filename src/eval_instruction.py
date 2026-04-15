import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data_setup import setup_dataloaders


def build_instruction_cfg(yaml_cfg):
    return {
        'model_name': yaml_cfg['teacher_model']['name'],
        'model': {
            'student': {
                'reduce_size': yaml_cfg['model'].get('student', {}).get('reduce_size', True),
                'size_reduction_factor': yaml_cfg['model'].get('student', {}).get('size_reduction_factor', 2),
            }
        },
        'data_path': yaml_cfg['data']['instruction_path'],
        'output_dir': yaml_cfg['output']['dir'],
        'max_length': yaml_cfg['data']['max_length'],
        'stride': yaml_cfg['data']['stride'],
        'batch_size': yaml_cfg['data']['batch_size'],
        'seed': yaml_cfg['training']['seed'],
        'is_instruction_tuning': True,
        'val_split_ratio': yaml_cfg['data'].get('val_split_ratio', 0.1),
        'val_stride': yaml_cfg['data'].get('val_stride'),
        'train_padding_side': yaml_cfg.get('tokenizer', {}).get('train_padding_side', 'right'),
        'generation_padding_side': yaml_cfg.get('tokenizer', {}).get('generation_padding_side', 'left'),
        'generation': yaml_cfg.get('generation', {}),
    }


def load_student_model(cfg, checkpoint_path, device):
    model_config = AutoConfig.from_pretrained(cfg['model_name'])
    if cfg['model']['student'].get('reduce_size', False):
        factor = cfg['model']['student'].get('size_reduction_factor', 1)
        model_config.num_hidden_layers = max(1, model_config.num_hidden_layers // factor)
        model_config.intermediate_size = max(1, model_config.intermediate_size // factor)

    model = AutoModelForCausalLM.from_config(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['student_model_state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys while loading checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys while loading checkpoint: {unexpected_keys}")
    model.eval()
    return model


def evaluate_instruction_model(model, dataloader, device, max_batches=None):
    total_nll = 0.0
    total_target_tokens = 0
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels'].clone()
            labels[batch['attention_mask'] == 0] = -100

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            logits = outputs.logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()

            batch_nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction='sum',
            )

            batch_target_tokens = int((shifted_labels != -100).sum().item())
            total_nll += batch_nll.item()
            total_target_tokens += batch_target_tokens
            processed_batches += 1

    if total_target_tokens == 0:
        raise ValueError("No target tokens found in evaluation set.")

    avg_ntp_loss = total_nll / total_target_tokens
    perplexity = math.exp(avg_ntp_loss)

    return {
        'ntp_loss': avg_ntp_loss,
        'perplexity': perplexity,
        'evaluated_batches': processed_batches,
        'evaluated_target_tokens': total_target_tokens,
    }


def generate_samples(model, tokenizer, dataloader, device, num_samples, generation_cfg, generation_padding_side):
    samples = []
    do_sample = generation_cfg.get('do_sample', False)
    generate_kwargs = {
        'max_new_tokens': generation_cfg.get('max_new_tokens', 80),
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': do_sample,
    }
    if do_sample:
        generate_kwargs['temperature'] = generation_cfg.get('temperature', 0.7)
        generate_kwargs['top_p'] = generation_cfg.get('top_p', 0.9)
    else:
        generate_kwargs['temperature'] = 1.0
        generate_kwargs['top_p'] = 1.0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = generation_padding_side

    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                for i in range(input_ids.size(0)):
                    if len(samples) >= num_samples:
                        return samples

                    sequence_length = int(attention_mask[i].sum().item())
                    target_positions = torch.nonzero(labels[i] != -100, as_tuple=False).flatten()
                    prompt_length = int(target_positions[0].item()) if target_positions.numel() > 0 else sequence_length

                    generation_input_ids = input_ids[i][:max(prompt_length, 1)].unsqueeze(0)
                    generation_attention_mask = torch.ones_like(generation_input_ids)
                    outputs = model.generate(
                        input_ids=generation_input_ids,
                        attention_mask=generation_attention_mask,
                        **generate_kwargs,
                    )

                    prompt = tokenizer.decode(input_ids[i][:prompt_length], skip_special_tokens=True)
                    ground_truth = tokenizer.decode(
                        input_ids[i][prompt_length:sequence_length],
                        skip_special_tokens=True,
                    )
                    prediction = tokenizer.decode(
                        outputs[0][generation_input_ids.shape[1]:],
                        skip_special_tokens=True,
                    )

                    samples.append(
                        {
                            'prompt': prompt,
                            'ground_truth': ground_truth,
                            'prediction': prediction,
                        }
                    )

        return samples
    finally:
        tokenizer.padding_side = original_padding_side


def main():
    parser = argparse.ArgumentParser(description="Evaluate an instruction-tuned checkpoint on held-out instruction data")
    parser.add_argument('--config', type=str, required=True, help='Path to instruction config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Phase 2 checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation outputs')
    parser.add_argument('--max_eval_batches', type=int, default=None, help='Optional limit on validation batches')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of qualitative samples to save')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    cfg = build_instruction_cfg(yaml_cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg['train_padding_side']

    _, val_loader = setup_dataloaders(cfg, tokenizer)
    model = load_student_model(cfg, args.checkpoint, device)

    metrics = evaluate_instruction_model(model, val_loader, device, args.max_eval_batches)
    samples = generate_samples(
        model,
        tokenizer,
        val_loader,
        device,
        args.num_samples,
        cfg['generation'],
        cfg['generation_padding_side'],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"instruction_eval_{Path(args.checkpoint).stem}_{timestamp}.json"
    report = {
        'checkpoint': args.checkpoint,
        'model_name': cfg['model_name'],
        'metrics': metrics,
        'samples': samples,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Saved instruction evaluation report to {output_path}")
    print(f"Held-out NTP loss: {metrics['ntp_loss']:.4f}")
    print(f"Held-out perplexity: {metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()
