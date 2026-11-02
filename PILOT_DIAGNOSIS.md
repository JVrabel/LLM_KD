# Why the 1B-token pilots score poorly: diagnosis (27 Oct 2026)

Scope: the eight finished Isambard pilots (0.6B and 1.7B dense Qwen3 students,
Python and Apollo medical text, next-token versus pure forward-KL distillation
from Qwen3.5-35B-A3B-Base, 1.00B tokens each). Everything below was measured
on the cluster today; raw results are in `logs/diag/` on the cluster and the
scripts are `scripts/isambard/diag_student.py` and `diag_forgetting.py`.

## Summary

1. **The trainer, the distillation loss, the teacher logits and the eval
   harness are correct.** No bug was found there. Distillation does what it
   should: it lowers held-out cross-entropy by 0.17 to 0.20 nats at both sizes
   and in both domains, and the 1.7B distilled Python student passes a strict
   superset of the HumanEval tasks its next-token twin passes.
2. **One real pipeline bug: the data is never shuffled.** `pack_shards.py`
   packs documents in file order. The Apollo corpus is three sub-corpora laid
   end to end, and the Python set is sorted by educational score. Every run
   therefore trained on one kind of text at a time, and the checkpoint sweep
   shows the models forgetting earlier slices by more than a nat while they
   train on later ones. This must be fixed before the next grid.
3. **The medical exam numbers measure a letter bias, not knowledge.** At
   0-shot the students answer "A" on 90 to 100% of questions and put only 4 to
   12% of their probability mass on the four letters. A cloze score, which
   rates the option text instead of the letter, shows a small but real
   in-domain signal: Apollo students reach 29 to 32% on medical MMLU and 28 to
   31% on MedQA against 25% for the Python students, ordered distilled above
   next-token and 1.7B above 0.6B.
4. **Beyond those two issues, the results are what one billion tokens buys.**
   phi-1-base needs about 51B tokens at 1.3B parameters for 29% HumanEval, and
   a 350M model on unfiltered Stack code saturates at 12% after about 200B
   tokens. Our 5.5 to 11.6% at 1B tokens sits on that curve. MMLU-style exams
   stay at chance for 1B-class models until trillions of tokens.
5. The Apollo question-answer file (660 MB) was never packed, so the medical
   students saw almost no exam-format text: 1535 of 2 million Apollo documents
   contain "Answer:".

## 1. What was checked and found correct

- **Teacher logit alignment.** On shard 0 of each domain the teacher's stored
  top-1 token equals the actual token at the same position for 78% of Python
  positions and 60% of Apollo positions. Both off-by-one alignments give under
  2%. Position 0 is empty as intended, every other position has 64 valid slots.
- **Top-64 coverage.** Mean probability mass in the stored top-64 is 98.3%
  (Python) and 95.0% (Apollo). The true token is in the top-64 at 98.5% and
  95.1% of positions. The leftover-mass bin in `topk_forward_kl` is therefore
  small and the loss is well-posed.
- **Distillation signal.** Cross-entropy on the same training batches is lower
  for the distilled run than for the next-token run by a steady 0.17 to 0.20
  nats throughout training, in all four pairs, even though the distilled runs
  never optimise cross-entropy. Held-out numbers agree (section 4).
- **Resume and schedule.** The learning rate is 6e-4 flat from step 300 to
  27465 and reaches 6.0e-5 at the end in every run. The resumed segments
  reproduce the pre-crash log values to four decimals. Tokens per step is
  32768 everywhere. Loss spikes occur at identical steps in the next-token
  and distilled runs on the same data (for example 18540 and 26260 on Apollo),
  so they are data-driven, not optimiser instability.
- **Eval harness.** HumanEval completions are well-formed Python with the
  right indentation and a return statement; they fail on logic. The letter
  tokens ` A` to ` D` are single tokens after "Answer:". Prepending EOS to the
  prompt, which matches how documents appear in training, changes no score by
  more than a point.
- **HumanEval passes are nested.** 1.7B distilled passes all 9 tasks the 1.7B
  next-token model passes plus 10 more; the union of the four Python models'
  passes is exactly the 1.7B distilled set. 18 of its 19 passes are among the
  60 easiest problems.

## 2. The data is not shuffled

`scripts/isambard/pack_shards.py` reads the JSONL in order and never shuffles
documents, and the trainer reads shards in order. What that order is:

**Apollo.** `filter_medical.py` writes the merged Apollo corpus as it was
given, and that corpus is a concatenation of sub-corpora. Documents per
4096-token sequence by shard:

| Shards | Docs per sequence | Content, from decoded samples |
|---|---:|---|
| 0 to 7 | 7.6 | textbook and reference text |
| 8 to 16 | 11 to 14.6 | short PubMed-abstract-like documents |
| 17 to 29 | 2.9 to 3.6 | long papers, guidelines, theses with reference lists |
| 30 to 32 | 3.9 to 8.6 | mixed, unused by the budget |

**Python.** `build_python_dataset.py` sorts by educational score descending,
then length. The first file has score 5.6 and is a guessing game; every
50000th line descends monotonically to score 3.0 at the end. Shard 0 holds 9
documents per sequence, shard 28 holds 3.5.

### Effect on training

Training cross-entropy per 2000 steps, 0.6B Apollo next-token (the distilled
run has the same shape):

| steps | 6k | 8k | 10k | 12k | 14k | 16k | 18k | 20k | 22k | 24k | 26k | 28k | 30k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CE | 2.52 | 2.54 | 2.52 | 2.44 | 2.39 | 2.61 | 2.83 | 2.78 | 2.71 | 2.72 | 2.69 | 2.59 | 2.55 |

The rise of 0.44 nats between 14k and 18k is the run crossing from abstracts
into long papers at shard 17, step 17408.

### Measured forgetting

Cross-entropy of successive checkpoints on 48 fixed sequences from each shard
(rows 4000 to 4047). Shards 31 and 33 were never trained on.

0.6B Apollo next-token:

| step | sh 0 | sh 4 | sh 8 | sh 12 | sh 16 | sh 20 | sh 24 | sh 28 | sh 31 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 2.91 | 2.96 | 3.87 | 3.71 | 3.68 | 3.94 | 3.85 | 3.95 | 4.04 |
| 5000 | **2.61** | 2.36 | 3.51 | 3.41 | 3.39 | 3.65 | 3.54 | 3.63 | 3.74 |
| 10000 | 3.66 | 3.56 | 2.91 | 2.61 | 2.61 | 3.58 | 3.53 | 3.55 | 3.79 |
| 16000 | **3.87** | 3.77 | 3.35 | 2.29 | 2.35 | 3.55 | 3.50 | 3.52 | 3.80 |
| 20000 | 3.35 | 3.27 | 3.04 | 2.57 | 2.51 | 2.87 | 2.76 | 2.82 | 3.49 |
| 25000 | 3.32 | 3.23 | 3.07 | 2.60 | 2.57 | 2.73 | 2.69 | 2.74 | 3.45 |
| 28000 | 3.28 | 3.20 | 3.00 | 2.59 | 2.57 | 2.73 | 2.56 | 2.71 | 3.40 |
| 30517 | 3.08 | 3.00 | 2.87 | 2.40 | 2.39 | 2.55 | 2.40 | 2.29 | 3.21 |

0.6B Apollo distilled:

| step | sh 0 | sh 4 | sh 8 | sh 12 | sh 16 | sh 20 | sh 24 | sh 28 | sh 31 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | **2.56** | 2.46 | 2.98 | 3.05 | 3.02 | 3.22 | 3.12 | 3.19 | 3.30 |
| 10000 | 3.09 | 3.04 | 2.52 | 2.43 | 2.43 | 3.15 | 3.08 | 3.11 | 3.28 |
| 16000 | **3.27** | 3.20 | 2.91 | 2.18 | 2.20 | 3.14 | 3.08 | 3.11 | 3.28 |
| 20000 | 2.93 | 2.88 | 2.60 | 2.38 | 2.36 | 2.67 | 2.56 | 2.61 | 3.01 |
| 25000 | 2.89 | 2.83 | 2.53 | 2.41 | 2.39 | 2.57 | 2.50 | 2.54 | 2.97 |
| 30517 | 2.69 | 2.65 | 2.38 | 2.24 | 2.23 | 2.41 | 2.28 | 2.25 | 2.78 |

0.6B Python next-token:

| step | sh 0 | sh 4 | sh 8 | sh 12 | sh 16 | sh 20 | sh 24 | sh 28 | sh 33 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | **1.91** | 1.67 | 2.15 | 1.95 | 2.13 | 1.99 | 1.80 | 2.03 | 2.34 |
| 10000 | 2.06 | 1.77 | 1.82 | 1.72 | 1.88 | 1.74 | 1.55 | 1.77 | 2.01 |
| 16000 | 2.20 | 1.83 | 1.91 | 1.58 | 1.79 | 1.65 | 1.44 | 1.67 | 1.97 |
| 20000 | 2.26 | 1.83 | 1.90 | 1.58 | 1.68 | 1.61 | 1.38 | 1.61 | 1.90 |
| 25000 | 2.24 | 1.81 | 1.88 | 1.59 | 1.72 | 1.57 | 1.41 | 1.61 | 1.78 |
| 28000 | **2.37** | 1.85 | 1.86 | 1.51 | 1.65 | 1.49 | 1.25 | 1.54 | 1.85 |
| 30517 | 2.17 | 1.68 | 1.70 | 1.38 | 1.49 | 1.36 | 1.15 | 1.21 | 1.64 |

The final Python model scores shard 0, the highest-rated files and the ones
seen first, at 2.17 nats against 1.21 on shard 28, which it saw last. The
unseen shard 33 sits at 1.64, so this is ordering, not difficulty.

While the Apollo model trains on shards 8 to 16 its loss on shard 0 rises from
2.61 to 3.87 nats; it only partly recovers later. Distillation damps the
swing but does not remove it. The Python set shows the same pattern at a
smaller amplitude. In a shuffled run every shard would improve monotonically.

Two consequences matter for the experiments:

- The last 10% of training, where the learning-rate decay produces most of the
  final gain (0.17 nats on Python), is spent entirely on the lowest-scored
  Python files and on the long-paper slice of Apollo.
- The textbook-style Apollo text, which is the slice closest to exam content,
  was seen 20000 steps before the end and partly forgotten.

Fix: shuffle documents before packing (a seeded permutation of the JSONL, or
a two-level shuffle: documents within large buckets, then bucket order). Also
consider shuffling sequence order within and across shards in the trainer.

## 3. The exam metric

### Letter scoring is a letter bias

The current score is which of ` A` to ` D` is most likely after "Answer:".
Diagnostics per model (0-shot medical MMLU; MedQA behaves the same):

| Model | Picks A/B/C/D | Mass on the 4 letters | Raw acc | Calibrated acc | Cloze acc |
|---|---|---:|---:|---:|---:|
| 0.6B Python next-token | 944/0/1/0 | 0.04 | 21.1 | 24.3 | 25.5 |
| 0.6B Python distilled | 872/24/39/10 | 0.06 | 21.2 | 23.5 | 25.3 |
| 1.7B Python next-token | 942/0/1/2 | 0.04 | 21.2 | 20.9 | 25.2 |
| 1.7B Python distilled | 931/1/9/4 | 0.07 | 21.7 | 25.5 | 25.0 |
| 0.6B Apollo next-token | 928/1/11/5 | 0.12 | 20.9 | 24.4 | 29.2 |
| 0.6B Apollo distilled | 775/91/77/2 | 0.40 | 22.2 | 27.6 | 29.8 |
| 1.7B Apollo next-token | 672/192/81/0 | 0.54 | 23.0 | 27.0 | 30.8 |
| 1.7B Apollo distilled | 819/26/95/5 | 0.64 | 20.8 | 25.4 | 31.9 |

Gold letters are 198/222/211/314, so always answering A scores 21%. The MedQA
"correct" counts of 352 to 360 reported for every student equal the 353 gold
answers that are A. Calibration (subtracting each letter's mean log-probability
over the test set) spreads the picks evenly but the accuracy stays at chance,
so the letter logits carry no question-specific information.

5-shot letter scores of 28 to 32% look better only because the shots teach
the format: letter mass rises to 0.91 to 0.96, picks move to C and D, and the
calibrated accuracy stays at 25 to 31%.

### Cloze scoring finds signal

Scoring each option text by its mean token log-probability after the
question (the method lm-eval calls `acc_norm`, and what `teacher_eval.py`
already does for MedQA):

| Model | MMLU cloze | MedQA cloze |
|---|---:|---:|
| Python students (all four) | 25.0 to 25.5 | 23.5 to 25.3 |
| 0.6B Apollo next-token | 29.2 | 27.8 |
| 0.6B Apollo distilled | 29.8 | 28.9 |
| 1.7B Apollo next-token | 30.8 | 28.4 |
| 1.7B Apollo distilled | 31.9 | 30.6 |

Standard error is about 1.4 points on MMLU and 1.2 on MedQA. The medical
students are 4 to 7 points above the Python students, and the ordering
(distilled above next-token, 1.7B above 0.6B) holds on both exams. This is
the in-domain signal the letter metric cannot see. The greedy samples agree:
the medical students continue "The patient presented with chest pain and"
with plausible clinical text and describe the mitral valve correctly at 1.7B,
while the Python students produce code on the same prompt.

Recommendation: report cloze accuracy, held-out cross-entropy and KL to the
teacher as the primary in-domain metrics at this scale, and keep letter
scoring for models that have learned the format.

## 4. What one billion tokens buys

Held-out cross-entropy, perplexity and KL to the teacher on 128 sequences of
shard 29 past the training cursor (identical rows for every model):

| Model | Python CE | Python ppl | Python KL | Apollo CE | Apollo ppl | Apollo KL |
|---|---:|---:|---:|---:|---:|---:|
| teacher (CE on covered tokens) | 0.69 | 2.0 | 0 | 1.43 | 4.2 | 0 |
| 0.6B Python next-token | 1.53 | 4.6 | 0.69 | 5.50 | 244 | 3.62 |
| 0.6B Python distilled | 1.35 | 3.9 | 0.51 | 4.81 | 122 | 2.96 |
| 1.7B Python next-token | 1.46 | 4.3 | 0.63 | 5.38 | 216 | 3.50 |
| 1.7B Python distilled | 1.29 | 3.6 | 0.45 | 4.64 | 103 | 2.79 |
| 0.6B Apollo next-token | 7.10 | 1215 | 6.19 | 2.49 | 12.1 | 0.77 |
| 0.6B Apollo distilled | 5.61 | 273 | 4.71 | 2.35 | 10.5 | 0.60 |
| 1.7B Apollo next-token | 7.11 | 1227 | 6.20 | 2.40 | 11.0 | 0.69 |
| 1.7B Apollo distilled | 5.45 | 232 | 4.54 | 2.24 | 9.4 | 0.50 |

Three readings:

- **The students are far from the teacher.** The best Python student is 0.6
  nats above the teacher on code; the best Apollo student is 0.8 nats above
  it on medical text. HumanEval and exam performance live in that gap.
- **Distillation is worth about 1.7 to 2 times the tokens.** The 0.6B
  distilled model beats the 1.7B next-token model on held-out loss in both
  domains. The gain is the same size at both scales, so it is not a small-model
  artefact.
- **Specialisation is strong and distillation leaks more.** Apollo students
  have perplexity 230 to 1230 on code against 3.6 to 4.6 for Python students.
  Distilled students have 2 to 5 times lower cross-domain perplexity than
  their next-token twins, because the teacher's distribution carries general
  knowledge into the in-domain tokens. This is a measurable leakage effect
  and worth tracking in the next grid.

Reference points for HumanEval, same greedy pass@1 protocol:

| Model | Params | Tokens seen | HumanEval |
|---|---:|---:|---:|
| ours, 1.7B next-token | 1.9B | 1B | 5.5% |
| ours, 1.7B distilled | 1.9B | 1B | 11.6% |
| 350M on unfiltered Stack (phi-1 paper) | 350M | ~200B | 12.2%, saturated |
| 350M on filtered Stack (phi-1 paper) | 350M | ~75B | 17.7% |
| StarCoderBase-1B | 1B | 1T | 15.2% |
| phi-1-base | 1.3B | 51B | 29% |
| phi-1 (plus 180M tokens of exercises) | 1.3B | 51B | 50.6% |

For exams: TinyLlama 1.1B scores 26% on MMLU after 3T tokens, and the
loss-threshold study of emergent abilities finds MMLU curves flat at chance
below roughly 500B tokens for models of this size. Qwen2-0.5B reaches 45% MMLU
only with many trillions of tokens plus distillation. At 1B tokens, chance on
letter-scored exams is the expected outcome regardless of training quality.

So the honest answer to "can models this small do it" is: not at this token
budget. A 1.7B student at 1B tokens sees 0.5 tokens per parameter. The
phi-1 recipe that motivates this project used 51B tokens, which at 15 steps
per minute would be 50 runs of 34 hours each on one GH200 for a single 1.7B
model. Hitting the phi-1 numbers needs either a much larger token budget,
multi-GPU training, or a smaller target (for example, a 0.6B student on 10 to
20B tokens, which is 5 to 10 allocations).

## 5. Secondary issues

- **Batch size.** 8 sequences of 4096 is 32k tokens per step at learning rate
  6e-4. Typical settings at this size use 0.5 to 1M tokens per step. The 10%
  decay phase removed 0.17 nats on Python, which is large and suggests a noisy
  stable phase. A sweep of batch 32 to 64 sequences with gradient accumulation
  is cheap to test on the next run, and would also raise GPU utilisation.
- **Weight decay** of 0.1 is applied to every parameter, including RMSNorm
  gains and the tied embedding. Common practice excludes norms and biases.
- **Throughput.** 0.6B runs at about 14k tokens per second and 1.7B at 8k,
  which is 6 to 9% of bf16 peak on a GH200. Master weights are fp32 and the
  248k-vocabulary projection in both losses runs in fp32 outside autocast.
  Running the projection in bf16 with fp32 softmax and using a larger batch
  should give a 2 to 3 times speedup. Not a correctness issue.
- **5-shot MedQA** truncates at 2048 tokens, which the handoff already notes.
  No prompt was skipped at 2048 in today's runs, so this did not bias the
  numbers above.
- **No exam-format text in training.** The Apollo QA file was not packed.
  Including it (decontaminated against MedQA and MMLU) would teach the format
  and is the cheapest way to make letter scoring meaningful.

## 6. Recommended order for the next iteration

1. Shuffle documents in `pack_shards.py` with a fixed seed and repack both
   domains. Keep the current shards for comparison.
2. Add the Apollo QA file to the medical pack after decontamination.
3. Switch the student exams to cloze scoring and add held-out CE and KL as
   standing metrics; keep letter scoring as a secondary number.
4. Re-run one pair (0.6B Apollo, next-token and distilled) on the shuffled
   data at the same budget to measure how much the ordering cost. About 40
   GPU hours.
5. Decide the token budget for the real grid from the phi-1 anchors above.
   Distillation at 1B tokens buys roughly a 2 times token multiplier; it does
   not replace the missing two orders of magnitude.

## Files

- Cluster results: `logs/diag/pilot_*.json` (held-out, exams, cloze, samples),
  `logs/diag/forget_*.json` (checkpoint sweeps).
- Scripts: `scripts/isambard/diag_student.py`, `diag_student.slurm`,
  `diag_forgetting.py`, `diag_forgetting.slurm` (cluster copies in `scripts/`).
- Sources: phi-1 paper https://arxiv.org/abs/2306.11644; emergent abilities and
  loss threshold https://arxiv.org/abs/2403.15796; TinyLlama leaderboard entry
  via https://www.analyticsvidhya.com/blog/2024/01/tinyllama-1-1b-size-doesnt-matter/;
  StarCoderBase-1B model card https://huggingface.co/bigcode/starcoderbase-1b;
  Qwen2 report https://arxiv.org/abs/2407.10671.

## 28 Oct 2026: shuffled rerun, 0.6B Apollo next-token

Same budget and schedule, sequences drawn in seeded random order across the
30 logit-covered Apollo shards plus the shuffled Apollo QA pack (33,346
sequences, about 12% of the pool). Job 7771850, checkpoint
`runs/pilot_0.6b_medical_apollo_ntp_shuffled/step_0030517.pt`.

Loss on the same 48 fixed rows per shard as the forgetting tables above
(`logs/diag/forget_pilot_0.6b_medical_apollo_ntp_shuffled.json`):

| step | sh 0 | sh 4 | sh 8 | sh 12 | sh 16 | sh 20 | sh 24 | sh 28 | sh 31 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | 2.84 | 2.85 | 2.47 | 2.70 | 2.67 | 3.13 | 3.01 | 3.08 | 3.52 |
| 10000 | 2.66 | 2.68 | 2.28 | 2.52 | 2.51 | 2.94 | 2.86 | 2.90 | 3.33 |
| 16000 | 2.57 | 2.58 | 2.13 | 2.42 | 2.43 | 2.85 | 2.74 | 2.79 | 3.24 |
| 20000 | 2.52 | 2.54 | 2.13 | 2.40 | 2.38 | 2.80 | 2.71 | 2.77 | 3.20 |
| 25000 | 2.50 | 2.46 | 2.09 | 2.37 | 2.37 | 2.80 | 2.69 | 2.75 | 3.17 |
| 30517 | 2.29 | 2.26 | 1.87 | 2.19 | 2.18 | 2.58 | 2.49 | 2.53 | 2.96 |
| unshuffled final | 3.08 | 3.00 | 2.87 | 2.40 | 2.39 | 2.55 | 2.40 | 2.29 | 3.21 |

Every shard improves monotonically at every checkpoint: the forgetting is
gone. The final model is 0.32 nats better on average over shards 0 to 28
(2.30 vs 2.62) and 0.25 nats better on shard 31, which neither run saw. The
late shards (20 to 28) are 0.1 to 0.25 nats worse than the unshuffled run,
which had just trained on them when it stopped.

Exams, answer-text scoring (per-token), 0-shot / 3-shot: MMLU 29.2 / 29.0
(unshuffled 29.2 / 29.5), MedQA 29.1 / 29.4 (unshuffled 27.8 / 27.5).
Letter scoring unchanged at chance. The loss gain does not reach MMLU; the
MedQA gain of about 1.5 points is where the exam-style QA pack would show.
Distilled twin (job 7771851) pending.

## 29 Oct 2026: shuffled rerun, 0.6B Apollo distilled

Job 7771851, checkpoint `runs/pilot_0.6b_medical_apollo_kd_shuffled/step_0030517.pt`,
same data and schedule as the shuffled next-token run, pure forward KL.
Final training KL 0.48 (unshuffled 0.64).

Per-shard loss on the fixed rows (`logs/diag/forget_pilot_0.6b_medical_apollo_kd_shuffled.json`):

| step | sh 0 | sh 4 | sh 8 | sh 12 | sh 16 | sh 20 | sh 24 | sh 28 | sh 31 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | 2.74 | 2.74 | 2.31 | 2.55 | 2.53 | 2.91 | 2.81 | 2.86 | 3.20 |
| 16000 | 2.49 | 2.50 | 2.05 | 2.30 | 2.30 | 2.66 | 2.55 | 2.58 | 2.93 |
| 30517 | 2.28 | 2.27 | 1.83 | 2.11 | 2.10 | 2.42 | 2.32 | 2.36 | 2.68 |
| unshuffled final | 2.69 | 2.65 | 2.38 | 2.24 | 2.23 | 2.41 | 2.28 | 2.25 | 2.78 |

Monotone at every shard again. Average over shards 0 to 28: 2.21 vs 2.39
unshuffled (0.18 nats); unseen shard 31: 2.68 vs 2.78 (0.10 nats). The loss
gain is smaller than for next-token (0.32 and 0.25) because the unshuffled
distilled run had forgotten less to begin with: the teacher's distribution
acts as a regulariser.

### The shuffled pair, side by side

| 0.6B Apollo, shuffled | next-token | distilled |
|---|---|---|
| mean loss, shards 0 to 28 | 2.30 | 2.21 |
| loss on unseen shard 31 | 2.96 | 2.68 |
| MMLU answer-text 0-shot / 3-shot | 29.2 / 29.0 | 31.1 / 32.8 |
| MedQA answer-text 0-shot / 3-shot | 29.1 / 29.4 | 30.6 / 32.6 |
| MMLU letter 0 / 5-shot | 21.1 / 29.7 | 23.7 / 30.1 |
| MedQA letter 0 / 5-shot | 27.5 / 27.7 | 26.5 / 27.7 |

Compared with the unshuffled distilled model (29.8 / 30.1 MMLU, 28.9 / 29.0
MedQA), shuffling plus the QA pack lifted the distilled student by 1.3 to
2.7 on MMLU and 1.7 to 3.6 on MedQA, more than it lifted the next-token
student. The 0.6B shuffled distilled model matches the unshuffled 1.7B
distilled model (31.9 / 30.6 at 0-shot). Few-shot examples, useless for
every unshuffled model, now add about 2 points for the distilled student.
Distilled beats next-token on every loss and answer-text measure, and the
gap on unseen text (0.28 nats) is larger than on seen text (0.09): the
distilled student generalises better. Letter scoring remains uninformative.
About 1.5% of MedQA and under 1% of MMLU test questions appear verbatim in
the QA pack; that bounds leakage at about one point and should be filtered
before the next grid.

## 30 Oct 2026: decontamination of the QA pack

`scripts/decontaminate_qa.py` removed every Apollo QA row whose question stem
(first 80 normalised characters) matches a MedQA-USMLE test or medical MMLU
test question: 30 of 417,236 rows, covering 26 distinct test questions (21
MedQA, 5 MMLU). The clean pack `data/packed/medical_apollo_qa_clean` has
33,344 sequences in 5 shards. The 0.6B shuffled results above were trained on
the contaminated pack; the leakage bound stays at about one point. The 1.7B
shuffled pair (next-token jobs 7867702/7867703, distilled 7867704/7867705)
trains on the clean pack and will be reported here when finished.

## 1 Nov 2026: shuffled rerun, 1.7B Apollo pair (clean QA pack)

Next-token jobs 7867702/7867703 (finished 1 Nov 03:36 UTC), distilled jobs
7867704/7867705 (1 Nov 17:19 UTC), each 24 h + about 11 h with
`--resume-latest` from step 21000. Data: Apollo 30 shards plus the
decontaminated QA pack, `shuffle: true`, same schedule as every other pilot.
Final training loss: next-token CE 2.10; distilled KL 0.41 (unshuffled 1.7B
distilled ended near 0.55), hard-token CE 2.04.

Per-shard loss on the fixed rows, final checkpoint
(`logs/diag/forget_pilot_1.7b_medical_apollo_{ntp,kd}_shuffled.json`):

| 1.7B shuffled, step 30517 | sh 0 | sh 4 | sh 8 | sh 12 | sh 16 | sh 20 | sh 24 | sh 28 | sh 31 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| next-token | 2.16 | 2.12 | 1.75 | 2.10 | 2.08 | 2.47 | 2.38 | 2.42 | 2.86 |
| distilled | 2.17 | 2.16 | 1.73 | 2.02 | 2.01 | 2.31 | 2.21 | 2.25 | 2.55 |

Both monotone at every shard and checkpoint (no forgetting). No per-shard
sweep exists for the unshuffled 1.7B pair, so the loss comparison is against
the shuffled 0.6B pair only.

### All four shuffled runs

| shuffled | 0.6B next-token | 0.6B distilled | 1.7B next-token | 1.7B distilled |
|---|---|---|---|---|
| mean loss, shards 0 to 28 | 2.30 | 2.21 | 2.19 | 2.11 |
| loss on unseen shard 31 | 2.96 | 2.68 | 2.86 | 2.55 |
| MMLU answer-text 0-shot / 3-shot | 29.2 / 29.0 | 31.1 / 32.8 | 31.6 / 31.0 | 31.0 / 32.8 |
| MedQA answer-text 0-shot / 3-shot | 29.1 / 29.4 | 30.6 / 32.6 | 29.4 / 31.0 | 31.2 / 31.8 |
| MMLU letter 0 / 5-shot | 21.1 / 29.7 | 23.7 / 30.0 | 22.8 / 24.9 | 22.0 / 23.3 |
| MedQA letter 0 / 5-shot | 27.5 / 27.7 | 26.5 / 27.7 | 24.4 / 28.0 | 27.1 / 27.7 |

Unshuffled 1.7B for reference (answer-text 0-shot / 3-shot): next-token
30.8 / 28.4 MMLU, 28.4 / 27.9 MedQA; distilled 31.9 / 31.2 MMLU, 30.6 / 28.4
MedQA.

Reading:

- Loss behaves as expected. Going from 0.6B to 1.7B lowers the mean loss by
  about 0.10 nats for both objectives, and distillation lowers it by a
  further 0.08 to 0.09. On unseen text the distilled advantage is 0.28 nats
  at 0.6B and 0.31 at 1.7B: distillation generalises better at both sizes
  and the effect does not shrink with size.
- Exams do not follow the loss. The 1.7B distilled model scores the same as
  the 0.6B distilled model (31.0 / 32.8 vs 31.1 / 32.8 on MMLU, 31.2 / 31.8
  vs 30.6 / 32.6 on MedQA). The forecast of 33 to 35 was wrong. With 945
  and 1273 questions one standard error is about 1.5 points, so every
  difference in the exam rows is within noise except the gap between
  shuffled and unshuffled few-shot scores. All eight Apollo students sit in
  a 29 to 33 band regardless of size, objective or ordering.
- Within that band, distilled is ahead of next-token in 11 of the 12
  shuffled answer-text comparisons (the exception is 1.7B MMLU 0-shot) and
  few-shot examples help the distilled students but not the next-token ones,
  so the ordering is consistent even if each gap is small.
- Conclusion for the write-up: at 1B tokens the exams have saturated at
  roughly 30 to 33 for any model in this grid; held-out loss, not the exams,
  is the measurement that separates the configurations. Larger absolute
  exam numbers need a larger token budget, as estimated in section 4.
