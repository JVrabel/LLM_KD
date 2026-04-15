
# LLM_KD — domain-focused knowledge distillation (“AI Savants”)

This repo is an experimental pipeline for **knowledge distillation (KD)** of a general-purpose teacher LLM into a **domain-focused student** trained on **domain-filtered data** (e.g., medical Apollo corpus). The working hypothesis is that a student trained primarily in-domain can retain strong in-domain capability while being much weaker outside the domain.

The notes and recent changes below are written to be “LLM-friendly”: they are intended to help a future agent (or teammate) understand *why this repo exists*, *what the current code does*, and *what changed recently*.

## Quickstart

### Train (KD / pretraining-style)

- Primary entrypoint: `python src/train_distr.py --config config/pretrain.yaml`
- Optional flags:
	- `--resume /path/to/checkpoint.pt`
	- `--ntp_only` (disable KD loss; train with NTP loss only)

On HPC, see `train.slurm` / `train_backup.slurm` (runs the same command inside an enroot container).

Important training controls in `config/pretrain.yaml`:

- `training.kd_ratio`: blend between KD and next-token prediction.
  - `0.0` = NTP-only baseline
  - `0.25`, `0.5`, `0.75` = recommended KD sweeps
  - `1.0` = pure KD diagnostic only
- `training.primary_val_metric`: metric used to pick `best_model.pt`
- `training.warmup_steps`: LR warmup length and default manual NTP warmup length
- `generation.*`: deterministic qualitative sample logging settings
- `model.student.size_reduction_factor`: set this to `> 1` to evaluate a genuinely smaller student

### Evaluate (MMLU)

Entrypoint: `python src/eval.py --config config/pretrain.yaml --output_dir src/outputs/eval ...`

Exactly one of the following must be provided:
- `--checkpoint /path/to/checkpoint.pt`
- `--use_base_model` (evaluate base HF model with student config)
- `--use_teacher_model` (evaluate full teacher)

Optional:
- `--medical_only`
- `--temp_dir /some/path` (where temporary exported models are written)

## Repo structure (high-signal files)

- `src/train_distr.py`: distributed KD training loop (DDP), checkpointing, periodic validation, sample generation.
- `src/train_instr_distr.py`: instruction-tuning / instruction distillation pipeline (in-progress).
- `src/data_setup.py`: dataset loading + preprocessing, including sliding-window tokenization and caching.
- `src/model_builder.py`: model construction for student + teacher and loss functions.
- `src/eval.py`: MMLU evaluation via lm-eval-harness.
- `src/eval_instruction.py`: instruction evaluation helpers (added in recent commit).
- `config/pretrain.yaml`: main pretraining / KD configuration.
- `config/instruction.yaml`: instruction-tuning configuration and data-format notes.
- `config/apollo_small_kd.yaml`: small Apollo smoke-test KD configuration.
- `config/apollo_small_ntp.yaml`: small Apollo smoke-test NTP-only configuration.
- `train.slurm`, `train_backup.slurm`: Metacentrum/HPC job scripts.

## Current experimental caveats

- Validation is now split at the JSONL-record level before tokenization. Older runs that used a post-tokenization split likely reported optimistic validation numbers because overlapping sliding windows could leak across train and validation.
- For checkpoint comparison, use `val_ntp_loss` / perplexity and deterministic sample generation. KD loss answers a different question than language-model quality.
- Phase 2 instruction tuning should only start after Phase 1 pretraining produces a trustworthy baseline on the cleaned split.

## Project context / motivation (from internal notes)

### “AI Savants” framing (why KD + filtered data)

Problem setup (high level):

- General-purpose (GP) models can have potentially-dangerous capabilities.
- Unlearning isn’t always reliable; capabilities may be recoverable after further tuning.
- Alternative approach: train a “narrow AI” by **whitelisting capabilities**:
	- Train from scratch on in-domain data, and/or
	- Distill in-domain behavior from a GP teacher using domain-filtered data.

Key questions:

- How much **capability leakage** remains after in-domain distillation?
- Can the student match in-domain performance while being weaker out-of-domain?
- Does instruction tuning change the optimal KD strategy vs base-model KD?

## Data formats

This repo uses JSONL sources.

- Pretraining/KD (sliding window): expects JSON objects with a `text` field.
- Instruction tuning: `src/data_setup.py` supports multiple formats (see `config/instruction.yaml` comments). The instruction path masks prompt tokens in `labels` so the model is supervised only on assistant response tokens.

Validation/data handling notes:

- Train/validation splitting is performed before sliding-window chunking so all windows from a source JSONL row stay on the same side of the split.
- `data.stride` / `data.val_stride` use the Hugging Face tokenizer meaning of `stride`: overlap between chunks, not step size. Set `data.val_stride: 0` for non-overlapping validation windows.
- Instruction-tuning metrics are not directly comparable to pretraining perplexity because prompt tokens are masked out during supervision.