# From-scratch distillation pilots: 0.6B and 1.7B students at 1B tokens

Marek Licko, 2 November 2026, updated 8 November 2026 (section 8). Repository: github.com/JVrabel/LLM_KD, branch `hpc_branch`.
Raw results: `reports/pilot_results/`; running notes: `PILOT_DIAGNOSIS.md`, `AGENT_HANDOFF.md`.

## 1. Question

Can a small dense model trained **from random initialisation** on a single domain, with
the teacher's token distribution as the target instead of the raw next token, reach a
useful fraction of the teacher's skill at a small token budget? The pilots compare pure
next-token training (NTP) against pure forward-KL distillation (KD) at two student sizes,
in two domains, at one billion training tokens each, on Isambard-AI (one GH200 per run).

## 2. Method

- **Teacher.** Qwen3.5-35B-A3B-Base (mixture of experts, 3B active). Its full-vocabulary
  log-probabilities were precomputed once per training sequence and stored as the top-64
  tokens per position (`scripts/isambard/precompute_logits.py`, 96% of gold tokens are
  covered, 12 GB per 33.5M-token shard, about 6.4 GPU-hours per shard).
- **Students.** Dense Qwen3-shaped transformers with the teacher's 248k vocabulary,
  random init: 0.6B (28 layers, width 1024, 0.44B non-embedding) and 1.7B (28 layers,
  width 2048, 1.4B non-embedding). Sequence length 4096.
- **Objective.** `kd_ratio` mixes next-token cross-entropy with forward KL
  KL(teacher || student) over the stored top-64 plus one bucket for the remaining mass
  (`src/train_kd.py`). The pilots use the two extremes, 0.0 and 1.0.
- **Schedule.** 30,517 steps of 8 x 4096 tokens (1.00B tokens, one pass), AdamW,
  peak LR 6e-4, 300 warm-up steps, constant LR to 90% of training, then linear decay to
  10%. Identical for every run. Sequences are drawn in a seeded random order across all
  shards (the first grid was unshuffled; see section 5).
- **Evaluation.** Medical: MMLU medical subset (945 questions, five subjects) and MedQA
  test (1,273), scored by the log-likelihood of the full answer text (letter scoring is a
  pure letter bias for models this small), 0-shot and 3-shot; one standard error is
  about 1.5 points. Python: HumanEval greedy pass@1. Held-out loss: cross-entropy on
  text the run never saw (Apollo shard 31, 48 fixed sequences; and shard 29 rows past
  the training cursor, where the teacher's stored logits allow a teacher comparison).

## 3. Data

| Pack | Content | Tokens | Teacher logits |
|---|---|---:|---|
| `medical_apollo` | ApolloCorpus English medical text (textbooks, guidelines, wiki, papers) | 1.08B (30 shards used) | yes |
| `medical_apollo_qa_clean` | Apollo exam-style question-answer pairs, exam test questions removed (8 MMLU/MedQA questions matched, all rows dropped) | 0.14B | yes |
| `medical` | the above plus 28M PubMed abstracts, exam-filtered | 9.62B | no |
| `python` | stack-edu Python, permissive licences, educational score >= 3 | 1.24B | yes |

The medical runs train on Apollo text plus the clean QA pack (about 12% of the pool);
the Python runs on the Python pack. Each run makes exactly one pass.

## 4. Results

**Held-out cross-entropy** (nats per token, lower is better). Unseen Apollo shard 31 for
the shuffled medical runs; shard 29 rows past the cursor for the teacher comparison and
the Python runs.

| | 0.6B NTP | 0.6B KD | 1.7B NTP | 1.7B KD | teacher |
|---|---:|---:|---:|---:|---:|
| Apollo shard 31, unseen | 2.96 | 2.68 | 2.86 | 2.55 | n/a |
| Apollo shard 29, unseen rows (first grid) | 2.49 | 2.35 | 2.40 | 2.24 | 1.74 to 1.9 |
| Python, unseen rows | 1.53 | 1.35 | 1.46 | 1.29 | 0.69 (covered tokens) |

The teacher's Apollo number is bracketed from its stored top-64 logits: 1.74 is the lower
bound, 1.8 to 1.9 the plausible value (`reports/pilot_results/teacher_ce_shard29_unseen.json`).

**Exams and code** (accuracy %, answer-text scoring, 0-shot / 3-shot).

| | 0.6B NTP | 0.6B KD | 1.7B NTP | 1.7B KD | teacher |
|---|---:|---:|---:|---:|---:|
| MMLU medical | 29.2 / 29.0 | 31.1 / 32.8 | 31.6 / 31.0 | 31.0 / 32.8 | 88.6 / 88.2 (0/5-shot) |
| MedQA | 29.1 / 29.4 | 30.6 / 32.6 | 29.4 / 31.0 | 31.2 / 31.8 | 82.6 / 83.1 |
| HumanEval pass@1 (Python runs) | 6.1 | 6.7 | 5.5 | 11.6 | 64.0 |

Chance on the exams is 25%. Python-only students score 25 to 28 on the medical exams.

Reading:

1. **Distillation works on the loss.** KD lowers held-out cross-entropy by 0.28 nats
   (0.6B) and 0.31 nats (1.7B) on unseen medical text and by 0.17 to 0.18 on code. The
   0.6B distilled student beats the 1.7B next-token student in both domains, so one
   objective change is worth more than tripling the parameters at this budget. The
   distilled students also leak less out of domain (2 to 5 times lower perplexity on
   the other domain), because the teacher distribution carries general knowledge.
2. **The exams have saturated at 29 to 33.** Every medical student, regardless of size,
   objective or data order, sits in that band; differences are within noise, though
   KD is ahead in 11 of 12 shuffled comparisons and few-shot examples help only the
   distilled students. HumanEval is the one benchmark that moved: 11.6% for the 1.7B
   distilled model against 5.5% for its next-token twin, and its passes are a strict
   superset of the twin's. These numbers match the literature for the token count: phi-1
   style models need about 50B tokens at 1.3B parameters for 29% HumanEval, and MMLU
   stays near chance for 1B-class models until hundreds of billions of tokens.
3. **The gap to the teacher is a data gap, not a teacher gap.** The teacher scores
   83 to 89 on the exams. The best student is 0.4 to 0.5 nats above the teacher on
   in-domain held-out text after seeing 0.5 tokens per parameter.

## 5. Issues found and fixed

- The packer wrote documents in corpus order and the trainer read shard by shard, so
  the first grid trained on one sub-corpus at a time and forgot earlier ones (loss on
  shard 0 rose by over a nat mid-run). Sequence-level shuffling was added and the
  medical grid rerun; losses are now monotone at every shard.
- The exam-style QA file was never packed; it is now packed and decontaminated.
- Letter-scored exams measured a letter bias (90 to 100% "A" at 0-shot); replaced by
  answer-text scoring.
- Budget: 795 GPU-hours used in total (teacher logits 480, 1.7B runs 199, 0.6B runs
  94, evaluations 23).

## 6. Scaling fit: what more tokens would buy

Fit `L(D) = E + B * D^-beta` to the smoothed training cross-entropy of each shuffled run
over its constant-learning-rate phase (0.03B to 0.9B tokens), then convert to unseen
shard 31 with the measured held-out offset and the measured anneal gain
(`scripts/isambard/fit_scaling.py`, `reports/pilot_results/scaling_fit.json`).

Check in the way Jakub suggested, fit on the first 0.45B tokens and predict 0.9B: a free
exponent (0.64 to 0.75) over-predicts the loss by 0.10 to 0.14 nats, the Chinchilla
exponent 0.28 under-predicts by 0.06 to 0.13, and beta = 0.40 lands within 0.01 nats
for both 1.7B runs (2.178 predicted vs 2.190 actual for KD). The table uses beta = 0.40.

| held-out loss on unseen shard 31, predicted | 1B (measured) | 3B | 9B | 20B | 50B | limit E |
|---|---:|---:|---:|---:|---:|---:|
| 1.7B KD | 2.55 | 2.44 | 2.36 | 2.31 | 2.28 | 2.19 |
| 1.7B NTP | 2.86 | 2.74 | 2.64 | 2.60 | 2.56 | 2.49 |
| 0.6B KD | 2.68 | 2.57 | 2.48 | 2.43 | 2.39 | 2.31 |

In teacher terms: the 1.7B distilled student is about 0.44 nats above the teacher on
unseen Apollo text at 1B tokens. Nine times the tokens are predicted to close about
half of that (0.23 nats), twenty times about 60%, fifty times about 70%. The fitted
asymptote of the student family (E = 1.79 nats on the training mix) is at the teacher's
level, which is the expected sanity check. There is no data in this grid that maps loss
to exam accuracy, because the exams never left the 29 to 33 band; by the loss-threshold
literature a 1.7B model needs several hundred billion tokens before MMLU-style exams
move far from chance, so a 9B-token run should be expected to land somewhere in the
mid to high 30s, not near the teacher's 88. A joint two-size Chinchilla fit (weakly
determined with two sizes) puts the compute-optimal point for a 9B-token 1.7B run at
2.3B parameters and 5.5B tokens, so the proposed run is near-optimal and slightly
over-trained, which is the right side for a deployable small model.

## 7. Conclusion and options

Within 1B tokens the project's method is validated at the level of the loss: pure
forward-KL distillation from the stored teacher distribution beats next-token training
by a margin worth more than tripling the model, at both sizes and in both domains, and
is the only configuration that moved a benchmark (HumanEval). It is not validated at
the level of the medical exams, which need far more tokens than any plausible
allocation here. The extrapolation says 9B tokens closes roughly half the loss gap to
the teacher and would leave the student well over twice as far from the teacher's exam
scores as the teacher is from perfect.

Cheapest next step that tests the extrapolation itself (about 170 GPU-hours, no new
teacher logits): continue the 0.6B and 1.7B distilled students for two more passes over
the existing logit-covered data (3.7B tokens total), with held-out loss and exams at
every 0.5B tokens. This triples the token range of the fit, checks the exponent, and
gives the first exam-versus-tokens curve. A 9B-token single-domain run with new PubMed
logits (about 1,000 GPU-hours, needs data-parallel training to fit the deadline) is
only worth it if that curve shows the exams starting to move.

## 8. Update, 8 November: both distilled students continued to 3B tokens

Jakub's suggested check: continue the two shuffled distilled students from
their 1B checkpoints to 3B tokens on the same pool (two more passes, fresh
permutations, learning rate re-warmed over 500 steps, then the same schedule),
evaluating every 0.5B tokens. Cost 113 GPU-hours.

| distilled, annealed finals | 0.6B at 1B | 0.6B at 3B | 1.7B at 1B | 1.7B at 3B |
|---|---|---|---|---|
| MMLU medical, answer-text 0 / 3-shot | 31.1 / 32.8 | 33.9 / 35.3 | 31.0 / 32.8 | 33.1 / 36.7 |
| MedQA, answer-text 0 / 3-shot | 30.6 / 32.6 | 30.3 / 31.9 | 31.2 / 31.8 | 32.0 / 32.5 |
| held-out loss, unseen shard 31 | 2.678 | 2.535 | 2.554 | 2.404 |
| section 6 prediction for 3B | | 2.570 | | 2.44 |

- The scaling fit held at both sizes and was conservative by 0.035 nats.
  Distillation follows an ordinary data power law here.
- MMLU moved for the first time: +2 to +4 points at both sizes, all four
  comparisons up, the two 3-shot gains beyond 1.5 standard errors. That is
  the first measured slope, about +3 MMLU points per tripling of tokens.
  MedQA did not move at either size.
- Mid-run checkpoints score below the annealed finals on exams even when
  their loss is lower; only annealed checkpoints are comparable.
- With the 15-fold token range now covered, the held-out trajectory fits a
  steeper exponent (0.54 to 0.61) with a higher plateau, but the anneal
  gain grows with tokens (0.16 nats at 1B, 0.21 at 3B). Predictions for the
  1.7B at 9B tokens range from 2.26 to 2.38 nats on the unseen shard. The
  gap to the teacher, about 0.44 nats at 1B on the same text, is 0.29 at
  3B and would be roughly 0.15 to 0.27 at 9B.
- Extrapolating the measured exam slope, a 9B-token 1.7B run would be
  expected around 36 to 40 on medical MMLU, 27B around 40 to 43. The
  teacher is at 88.6, so the "within 2x of the teacher" mark (44) needs
  tens of billions of tokens of this corpus, or data with denser knowledge
  per token.
