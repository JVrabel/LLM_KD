# AI Savants: review of the 1B runs and plan v2 (two domains, small students)

All GPU-hour figures are estimates assuming about 300 TFLOP/s effective per GH200 (≈30% of bf16 peak). They should be checked against measured throughput before committing the budget. Benchmark numbers for third-party models are vendor-reported.

## Summary

- **The 1B runs produced no usable signal.** Every student scores at chance on medical MMLU, so there is no validated recipe yet to carry over.
- **The main causes are the experimental setup, not the KD idea:** a from-scratch 1B student on ~1.4B tokens, a weak 1B teacher, a 512-token training context, and text cleaning that removed all newlines. The training code also has several bugs, one of which left a full epoch training at learning rate 0.
- **Goal of v2:** break the scaling curve for narrow specialists. A 0.6–1.7B student trained on a few billion high-quality domain tokens should match general models several times its size on its domain, using well under 1% of their training tokens. The approach follows "Textbooks Are All You Need" (phi-1): quality over quantity.
- **Plan (~1,650 GPU-hours including contingency):**
  - Two domains: medicine and code (Python).
  - Teachers: Qwen3.5-35B-A3B-Base for distillation, Qwen3.8-27B for synthetic data and instruction tuning. They share the same tokenizer.
  - Students: small dense Qwen3-architecture models (0.6B and 1.7B) trained from scratch, each with a no-KD baseline.
  - Data per domain: ~1–2B tokens of classifier-filtered real text plus ~1B tokens of synthetic textbooks and exercises.
  - Evaluation: in-domain benchmarks, cross-domain leakage (the medical student on code and vice versa), and reference general models of the same and larger sizes.

## 1. What the existing runs show

Medical MMLU results (5-shot, five subjects, 945 questions):

| Model | Accuracy |
|---|---|
| Llama-3.2-1B-Instruct (teacher, Phase 2) | 48.6% |
| Llama-3.2-1B (teacher, Phase 1) | 34.2% |
| Full-size pure-KD student, 10 epochs (`best_model-001.pt`) | 25.9% |
| Instruction-tuned students (3 runs) | 23.2–25.0% |
| Reduced 2× students (KD, pure KD, NTP baseline) | 22.4–23.5% |

Chance is 25%, and the standard error over 945 questions is about ±1.4 points, so no student is distinguishable from chance. On in-domain text the best student is reasonably close to the teacher (validation KL ≈ 0.15 nats/token, perplexity 11.25). That suggests MMLU is measuring something the students were never trained for, not missing medical knowledge.

**Why:**

1. **MMLU is at its floor for this setup.** The student is trained from scratch (1.24B parameters) on ~1.4B unique tokens from the Apollo corpus. Open 1B models trained from scratch on trillions of tokens (TinyLlama, Pythia) also score around 25–26% on MMLU. Llama-3.2-1B is above chance mainly because it was distilled from 8B/70B models over 9T tokens.
2. **Context mismatch.** Training used 512-token windows. Every 5-shot `professional_medicine` prompt is longer than that (median 1,067 tokens), and that is the one subject where all students score *below* chance.
3. **Destructive text cleaning.** `clean_text` collapses all whitespace and removes all parentheses (both are present in ~80% of Apollo documents). The Phase 1 student never saw a newline token, which the MMLU prompt format depends on.
4. **Weak, mismatched teacher.** The 1B teacher is only ~9 points above chance on medical MMLU, and it was run in 4-bit during training, while its reported score is from full precision.
5. **No out-of-domain evaluation.** The hypothesis is "strong in-domain, weak out-of-domain", but leakage was never measured.

**Bugs found:**

- **Resume breaks the LR schedule** (`train_distr.py`). The NTP warmup reruns on every resume, and the restored scheduler step count no longer matches the recomputed schedule length after a change in GPU count or epochs. Confirmed in `logs/train_20260408_001152.log`: one whole epoch (~13 h of compute) ran at `lr=0`.
- **Instruction-tuning losses are logged 4× too small** (`train_instr_distr.py`). They are already divided by the gradient-accumulation steps, so train-versus-validation comparisons look like overfitting when they aren't.
- **Eval can silently score the teacher** (`eval.py`). It loads pretrained teacher weights and then overlays the checkpoint with `strict=False`; any key mismatch would leave the teacher's weights in place.
- **Smaller issues:** fp16 autocast instead of bf16; leftover gradient accumulation carried across epochs; no end-of-sequence token between documents; heavy padding; 2.5-hour full validation passes.

**Why not just rerun the old recipe at 8B (e.g. Llama-3.1-8B → 8B student):** the current trainer replicates the full model on every GPU (DDP), and an 8B model with AdamW state needs ~130 GB, more than a 96 GB GPU. Apollo alone is ~0.2 tokens per parameter for an 8B student, so it would very likely also end at chance. A new trainer and new data are needed either way.

## 2. Teachers

| Role | Model | Why |
|---|---|---|
| Distillation teacher | Qwen3.5-35B-A3B-Base | Pretrained-only (undistorted next-token distribution); mixture-of-experts with 3B active parameters, so scoring text is cheap |
| Synthetic data, instruction tuning | Qwen3.8-27B | Strongest open model of its size, especially for code; post-trained only, no base checkpoint |
| Fallback | Llama-3.1-8B (+ Instruct) | Plain dense architecture and the simplest setup; much weaker; gated licence (needs an HF token) |

All Qwen3.5 and Qwen3.8 models use the same vocabulary, merges and pre-tokenizer (I compared the files); Qwen3.8 only adds special tokens such as `<think>`. One student tokenizer therefore works with every Qwen teacher.

Quality comparison (post-trained versions; Qwen in thinking mode, so not directly comparable with Llama, but the gap is large):

| Benchmark | Llama-3.1-8B-Instruct | Qwen3.5-9B | Qwen3.5-35B-A3B | Qwen3.8-27B |
|---|---|---|---|---|
| MMLU-Pro | 48.3 | 82.5 | 85.3 | not reported |
| SuperGPQA | not reported | 58.2 | 63.4 | not reported |
| GPQA Diamond | 30.4 | 81.7 | 84.2 | 89.2 |
| LiveCodeBench v6 | not reported | 65.6 | 74.6 | 90.3 |

**Smaller teachers do not save compute.** What matters for cost is active parameters per token. Qwen3.5-9B is dense, so precomputing its logits costs ~3× more than for the 35B-A3B (roughly 170 vs 60 GPU-h per 10B tokens, theoretical). Its only practical advantage is memory (19 vs 67 GB): it could run online next to the student. Teacher logits are a one-off cost of ~5–10% of the budget either way.

**Teacher comparison before committing (~20 GPU-h):** Qwen3.5-35B-A3B-Base, Qwen3.5-9B-Base and Llama-3.1-8B on held-out medical and code text (perplexity), MedQA / MedMCQA / PubMedQA by likelihood, and HumanEval / MBPP few-shot.

**On running the teacher on the CPU:** not worthwhile. The GH200's fast CPU–GPU link helps with memory, not compute; 72 ARM cores are 100× or more too slow for the teacher's forward pass, and the teacher fits on the GPU anyway.

## 3. Students

**Plain dense Qwen3 architecture** (`Qwen3ForCausalLM`: grouped-query attention, RoPE, SwiGLU, QK-norm), trained from scratch with the Qwen3.8 tokenizer, 4,096-token context, tied embeddings.

| Shape | Width / layers / heads (Q/KV) | Params with 248k vocab | Size in bf16 | Cost per 10B tokens seen |
|---|---|---|---|---|
| Qwen3-0.6B | 1024 / 28 / 16-8 | 0.69B | 1.4 GB | ~40 GPU-h |
| Qwen3-1.7B | 2048 / 28 / 16-8 | 1.92B | 3.8 GB | ~105 GPU-h |
| Qwen3-4B (only if the curve looks promising) | 2560 / 36 / 32-8 | 4.27B | 8.5 GB | ~240 GPU-h |

- Small students fit the goal (running on small hardware) and make a size/data grid affordable. A student the same size as the 35B mixture-of-experts teacher would not be simpler.
- Cost to accept: with the 248k vocabulary, ~37% of the 0.6B model is embeddings. A smaller vocabulary would require cross-tokenizer distillation, which is much more complex.
- I'd avoid the teachers' hybrid Gated DeltaNet architecture: it needs custom kernels (compiled from source on ARM), and the architecture isn't the research variable.

## 4. Data (per domain: ~1–2B tokens filtered real + ~1B synthetic)

Following phi-1: train a quality classifier on ~50k documents labelled by Qwen3.8-27B ("is this textbook-quality material for learning the domain?"), keep the top slice of the real data, and add synthetic textbooks and exercises. For reference, phi-1 (1.3B) reached 50.6% on HumanEval with ~7B unique tokens: ~6B filtered web code plus ~1B synthetic textbooks.

**Medicine**

| Source | Status | Rough size |
|---|---|---|
| PubMed abstracts (2026 baseline) | On Isambard, all 1,334 files MD5-verified | ~5–7B tokens (not yet counted) |
| Apollo corpus + QA | On Isambard, checksums verified | ~1.4B tokens |
| Clinical guidelines (`epfl-llm/guidelines`, open portion) | On Isambard | small, high quality |
| PMC Open Access full text | Pending | tens of billions; keep licence subsets separate |
| DailyMed / openFDA labels, ClinicalTrials.gov | Pending | ≲1B |

**Code (Python)**

| Source | Notes |
|---|---|
| Stack-Edu Python (`HuggingFaceTB/stack-edu`) | Educational-quality subset of The Stack v2, already classifier-filtered; file contents are fetched from Software Heritage |
| The Stack v2 Python (`bigcode/the-stack-v2`) | Larger pool to filter ourselves if Stack-Edu is not enough; permissive licences only |
| Python documentation, PEPs, Stack Exchange dump | Explanatory text (Stack Exchange is CC BY-SA) |

**Synthetic data (Qwen3.8-27B via vLLM on Isambard):**
- Medicine: textbook-style chapters seeded by MeSH topics, worked clinical cases, and exam-style questions with explanations. These fill the gap between PubMed's abstract style and benchmarks like MedQA.
- Code: textbook sections with examples, and exercises with unit tests that are executed, keeping only solutions that pass (as in phi-1's CodeExercises).
- Rough cost: 30–150 GPU-h per 1B tokens depending on throughput, to be measured first.
- Frontier-model APIs: only for labelling and quality checks, not bulk generation (cost, terms of service forbidding training competing models, and general capabilities contaminating the leakage measurement).

**Hygiene:** hold out in-domain and out-of-domain eval text before training; deduplicate; decontaminate against all eval sets (Apollo's exam questions may overlap MedQA or MMLU, PubMedQA is built from PubMed abstracts, HumanEval/MBPP solutions circulate widely in code corpora). Skip MIMIC for now (credentialed access, restrictive data-use agreement).

## 5. Experiments and evaluation

**Training:**
- Phase 1: precompute the base teacher's top-64 log-probabilities once with vLLM (`prompt_logprobs`, `max_logprobs=64`; ~384 bytes/token, ~4 TB per 10B tokens in `$PROJECTDIR`) and reuse them for every student. Loss: forward KL on the top-64 plus cross-entropy on the true next token, starting at T=1.
- Phase 2 (instruction tuning, smaller): Qwen3.8-27B with its chat template and thinking disabled.
- Hyperparameters: the old ones (LR 1e-4, ~16k tokens per step, linear decay) do not transfer. AdamW (β2=0.95, weight decay 0.1), warmup–stable–decay schedule, batch size in tokens; tuned in the pilot.

**Grid per domain (~6B tokens seen per run, i.e. 2–3 epochs):**

| Variable | Values |
|---|---|
| Student size | 0.6B, 1.7B |
| Objective | KD, no-KD (plain next-token prediction) |
| Data | filtered real only; filtered real + synthetic |

Plus, for medicine, one quantity-versus-quality run: 1.7B with KD on all real medical text (~8B tokens) versus the filtered ~1–2B. That tests the "textbooks" claim directly.

**Evaluation:**
- **Medicine:** MedQA, MedMCQA and PubMedQA by likelihood; medical MMLU 0-shot and 5-shot; perplexity on held-out medical text.
- **Code:** HumanEval(+) and MBPP(+) pass@1 via EvalPlus; perplexity on held-out code.
- **Leakage:** each specialist on the other domain's benchmarks; ARC-Easy, PIQA, HellaSwag and non-medical MMLU; general web-text perplexity.
- **Reference points for the scaling claim:** general models evaluated with the same harness, e.g. Qwen3-0.6B / 1.7B (~36T tokens), Llama-3.2-1B (~9T), Llama-3.1-8B (~15T), the teachers, and phi-1 for code. Plot in-domain score against training tokens and compute.

**Success criterion:** a 1.7B specialist matches or beats a general model of ≥4B parameters on its domain while staying clearly weaker out of domain, with KD beating the no-KD baseline.

## 6. Budget (first iteration)

| Step | GPU-hours |
|---|---|
| Data processing: classifier labelling, filtering, deduplication, eval sets | ~100 |
| Teacher comparison | ~20 |
| Pilot: 0.6B on a ~1B-token medical slice; T ∈ {1, 2}, KD ratio ∈ {0.5, 1.0}, LR | ~40 |
| Synthetic data, ~1B tokens per domain | ~200 |
| Teacher top-64 logits, ~12B tokens across both domains | ~100 |
| Training grid, both domains + quantity-versus-quality run | ~800 |
| Phase 2 instruction tuning for the best students | ~60 |
| Evaluation, including reference models | ~60 |
| Contingency (~20%) | ~270 |
| **Total** | **~1,650** |

**Go / no-go after the pilot:** run the full grid only if the 0.6B student is above chance on at least one in-domain benchmark and KD beats the no-KD baseline on in-domain perplexity.

## 7. Code changes needed

ARM itself matters little. PyTorch and vLLM have ARM builds; bitsandbytes is uncertain on ARM, so teachers run in bf16 instead of 4-bit. Most changes come from the new teachers, the second domain and the new data.

1. **Data pipeline:**
   - Drop `clean_text`; normalize Unicode only.
   - Domain-agnostic ingestion (PubMed XML, JSONL, code files), quality classifier, deduplication and decontamination.
   - Pack documents with end-of-sequence tokens into 4,096-token sequences, stored as `uint32` memmap shards.
2. **Synthetic data generation:** vLLM jobs with prompt templates per domain; a sandboxed test runner for code exercises.
3. **Teacher-logit extraction:** a vLLM job writing top-64 shards aligned with the token shards.
4. **New trainer (`train_kd.py`):**
   - FSDP2 in bf16; multi-node via `torchrun` / Slurm.
   - A top-k KD loss, plus chunked cross-entropy over the 248k vocab.
   - Step-based sharded checkpoints every ~45 minutes, with exact resume of model, optimizer, scheduler, data position and RNG state.
5. **Phase 2 trainer:** Qwen3.8-27B teacher, chat template, fixed loss logging.
6. **Evaluation:** `strict=True` checkpoint loading; lm-eval for medicine and general tasks, EvalPlus for code, held-out perplexity sets, reference-model runs.

## 8. Isambard status

- SSH and Slurm work non-interactively; the clifton certificate must be renewed daily (`clifton auth`).
- Login nodes kill background processes on logout (`KillUserProcesses=yes`), so long tasks run as Slurm jobs. Compute nodes have internet access.
- Layout in `/projects/u6wc/savants/` (group-writable for the project): `models/` holds Qwen3.5-35B-A3B-Base (67 GiB, verified) and a link to the existing Qwen3.8-27B copy; `data/raw/` holds PubMed baseline (51 GB), Apollo, guidelines and the eval sets.
- Download scripts: `scripts/isambard/` in the repo (resumable, checksum-verified).

## Open questions

1. Is code the right second domain? Math is the main alternative (verifiable, but harder for small models).
2. Should Phase 2 (instruction tuning) be in this iteration, or only Phase 1 specialists?
3. Which API credits are available for labelling and quality checks?
4. Any target hardware for the final student (e.g. a phone, a laptop CPU)? That would fix the maximum student size.
