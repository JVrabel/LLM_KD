# Handoff: AI Savants distillation pilots on Isambard-AI Phase 2

Read this before touching the cluster or the trainer. The git checkout and the
cluster copy can differ. Jobs run the cluster copy. As of 27 Oct 2026 the
eight pilot runs and all their evals are finished, no `config/pilot_*.yaml`
carries a `resume:` key, and the laptop and cluster copies of `src/`,
`scripts/` and `config/` match. The diagnosis of the weak scores is in
`PILOT_DIAGNOSIS.md`; the section "27 Oct 2026: shuffling fix" below says
what changed and what is waiting for a decision.

Local repo: `/Users/marek/Projects/LLM_KD`, branch `hpc_branch`. Uncommitted.
Cluster project: `/projects/u6wc/savants` (same tree as
`/lus/lfs1aip2/projects/u6wc/savants` and `$PROJECTDIR/savants`).
Sync by `scp` into `src/`, `config/`, and `scripts/` on the cluster. There is
no automatic deploy. Cluster scripts live in `scripts/`, not `scripts/isambard/`.

## Access

```bash
clifton auth          # browser login; writes a 12-hour SSH certificate
ssh u6wc.aip2.isambard
```

- User `marek.u6wc`, account `brics.u6wc`, QOS `normal`, partition `workq`.
- Certificate: `/Users/marek/Library/Caches/clifton/u6wc.aip2.isambard-cert.pub`.
- SSH config: `~/.ssh/config` includes `~/.ssh/config_clifton`. Host
  `u6wc.aip2.isambard` uses that certificate, identity `~/.ssh/id_ed25519`,
  and jumps through `ai.login.isambard.ac.uk`.
- `Permission denied (publickey)` with the certificate file unchanged means
  the cert expired. `clifton auth` must finish in a terminal on this Mac.
  A browser login that never returns to that process does not update the file.
- Docs: https://docs.isambard.ac.uk
- Status: https://status.isambard.ac.uk
- Support: https://support.isambard.ac.uk

### Scheduler limits

- This account cannot request more than 24:00:00. A test of 48h and 72h was
  rejected with `QOSMaxWallDurationPerJobLimit`. Partition `workq` shows
  `MaxTime=UNLIMITED`, but the account still caps a job at 24 hours.
  `restricted48` exists and is not granted to this account.
- Training and logit jobs must checkpoint and be resubmitted. The trainer
  resumes only when the yaml contains `resume: /path/to/step_XXXXXXXX.pt`.
  It does not pick the newest checkpoint by itself.
- Slurm `Requeue=1`. After `NODE_FAIL` it reruns the same script and therefore
  the yaml's `resume:` path. On 23 Oct 2026 a node failure restarted runs from
  step 21000 or from step 0 and nearly overwrote newer checkpoints. Before any
  resume, list `runs/<name>/step_*.pt` and point `resume:` at the newest
  full-size file.
- One 1.7B checkpoint is about 23 GB. A 0.6B checkpoint is about 8.3 GB.
  Same size as its neighbours means the file was not torn by the node failure.
- Logit jobs need `#SBATCH --nodes=1` and `--gpus-per-node=2`. Without that,
  Slurm can split the two GPUs across hosts and vLLM tensor-parallel fails.
- `/local/user/$UID` is not writable. Eval and logit scripts must set
  `TMPDIR`, `TORCHINDUCTOR_CACHE_DIR`, and `TRITON_CACHE_DIR` under
  `/projects/u6wc/savants/job-cache/$SLURM_JOB_ID`. Training happened to work
  without that. Evals died with `PermissionError` until it was set.
- Controllers: `gracehopper-slurmctld` (`10.242.255.210`) and
  `gracehopper-slurmctld-backup` (`10.242.255.211`). Both were unreachable
  from the evening of 25 Oct 2026 until the afternoon of 26 Oct. The status
  page said "No Known Issues". Check with `scontrol ping` before submitting.
- Posted maintenance, not the cause of that outage: network blips on
  26 Oct 2026 10:00–14:00 UTC. Cooling shutdown of Phase 2 compute nodes on
  4 Nov 2026 08:00–17:00 UTC (provisional). Login nodes stay up; jobs queue
  until nodes return.

## What this project is

From-scratch dense Qwen3 students, trained on about 1.00 billion tokens, with
an optional frozen teacher. The teacher is `Qwen3.5-35B-A3B-Base` (about 35B
total, 3B active, hybrid Gated DeltaNet). Tokenizer and teacher weights:
`/projects/u6wc/savants/models/Qwen3.5-35B-A3B-Base`. Vocabulary 248320.
EOS id 248044.

Students are `Qwen3ForCausalLM` with tied embeddings, defined in
`src/train_kd.py` as `STUDENTS`:

| Key | hidden | intermediate | layers | heads / kv | Measured parameters |
|---|---:|---:|---:|---|---:|
| `0.6b` | 1024 | 3072 | 28 | 16 / 8 | about 0.70B |
| `1.7b` | 2048 | 6144 | 28 | 16 / 8 | 1.918B |
| `4b` | 2560 | 9728 | 36 | 32 / 8 | about 4.27B, not trained |

`4b` has only `config/pilot_4b_smoke.yaml` (2 steps). Do not treat it as a
planned run. Batch 8 of the 1.7B model fits on one GH200 (smoke job 7598686).

Two domains:

- Python code: `data/packed/python`, logits `data/logits/python`. Shard 29
  is done. The 1B-token budget does not need past shard 29.
- Apollo medical text only: `data/packed/medical_apollo`, logits
  `data/logits/medical_apollo`. Shards 0–29 are done
  (`DONE {"shards": 30, "topk": 64}` from job 7598044). Packed data has
  shards 0–32; do not score 30–32 for this budget.

Packed shards hold 8192 sequences. The budget is 30517 steps times 8
sequences = 244136 sequences, which ends inside shard 29.

## Loss

`src/train_kd.py` is the trainer that matters. Ignore `src/train_distr.py`
for these pilots.

`kd_ratio` is a mixture weight, not an extra term:

- `0`: next-token cross-entropy only. `logit_dir` is not read.
- `1`: pure forward KL. Cross-entropy is not in the optimized loss. On log
  steps it is still computed under `no_grad` and printed as `ce`.
- Between 0 and 1: `(1 - kd_ratio) * ce + kd_ratio * kl`.
- Values outside `[0, 1]` abort. `temperature` must be `1.0` whenever
  `kd_ratio > 0`. The stored values are log probabilities, not logits, so
  they cannot be softened.

`topk_forward_kl` uses the teacher's stored top-64 log probabilities as
full-vocabulary probabilities. They are not renormalized to 1. Sentinel
slots near `-1e4` are ignored. Student log probabilities come from a softmax
over the whole vocabulary. The leftover teacher mass, `1 - sum(p_topk)`, is
one extra bin, and the student's bin is whatever probability it placed
outside those 64 ids. That residual bin is required. Matching only the 64
tokens makes the optimum dump all leftover mass onto those 64.

An earlier Python KD job used `CE + 1.0 * KL` with a 64-way renormalized
softmax. It was cancelled after a few hundred steps. That checkpoint is
`runs/pilot_0.6b_kd_cekl/step_0001000.pt`. Do not evaluate it as the
distillation result. The real run is `runs/pilot_0.6b_kd/step_0030517.pt`.

Checkpoints are version 2 and store `objective` (`next_token`, `forward_kl`,
or `mixed`), `model_size`, `model_config`, optimizer, data cursor, and RNG.
Eval scripts rebuild the architecture from that metadata.

## Shared schedule

All full runs: seed 42, 30517 steps, warmup 300, decay from step 27465,
learning rate `6e-4`, floor 0.1, batch 8, AdamW betas `(0.9, 0.95)`, weight
decay 0.1, log every 20, save every 1000. One GPU, 16 CPUs, 24-hour Slurm
limit. 0.6B runs take about 19 hours (about 26 steps/minute) and finish in
one allocation. 1.7B runs take about 15 steps/minute, about 34–35 hours, so
they need a second allocation.

Launch training from the cluster project directory:

```bash
cd /projects/u6wc/savants
sbatch --job-name=NAME \
  --output=/projects/u6wc/savants/logs/NAME_%j.log \
  --export=ALL,CONFIG=/projects/u6wc/savants/config/FILE.yaml \
  scripts/train_pilot.slurm
```

The script loads `cuda/12.6` and `venv-train` (PyTorch 2.10.0+cu126).
Logit precompute uses `venv-vllm126`, vLLM 0.17, two GPUs, and GCC 13.4 at
`/projects/u6wc/savants/gcc13` (`aarch64-conda-linux-gnu-g++`). FlashInfer
rejects other GCC versions. `HF_HUB_OFFLINE=1`.

## Finished runs

Times below are Europe/Madrid. Through 24 Oct 2026 that was CEST (UTC+2).
Sacct `End` fields were UTC.

### 0.6B Python next-token — done

- Config `config/pilot_0.6b.yaml`, no resume key.
- Checkpoint `runs/pilot_0.6b_ntp/step_0030517.pt`.
- Final loss about 1.24.
- HumanEval: 10/164 = 6.1%. Result file on the cluster:
  `logs/student_humaneval_results.json`. Do not overwrite it.
- Same checkpoint on medical exams, which it was not trained for:
  MMLU 0-shot 21.1% (199-ish/945; recorded 21.06%), 5-shot 32.1%;
  MedQA 0-shot 27.7%, 5-shot 24.5%.
  Cluster file `logs/student_mmlu_results.json`. Do not overwrite it.

### 0.6B Python pure distillation — done

- Job 7580064. Finished 21 Oct 2026 about 12:02, 19 h 3 min.
- Checkpoint `runs/pilot_0.6b_kd/step_0030517.pt`.
- Step 30500: KL 0.45, logged hard-token loss 1.09.
- HumanEval job 7598054: 11/164 = 6.7% (`pass_at_1` 0.0671).
  File `logs/python_kd_humaneval.json`.
- One extra pass over the next-token baseline is inside the noise. 164 tasks,
  standard error around 2 points near this rate.

### 0.6B Apollo next-token — done

- Job 7578645. Finished 21 Oct 2026 about 11:13, 19 h 21 min.
- Checkpoint `runs/pilot_0.6b_medical_apollo_ntp/step_0030517.pt`.
- Step 30500 loss 2.60.
- Medical eval job 7598055, file `logs/apollo_ntp_mmlu.json`:

| | MMLU 0-shot | MMLU 5-shot | MedQA 0-shot | MedQA 5-shot |
|---|---:|---:|---:|---:|
| Python 0.6B, no medical text | 21.1% | 32.1% | 27.7% | 24.5% |
| Apollo 0.6B next-token | 21.0% (198/945) | 28.8% (272/945) | 27.7% (352/1273) | 27.8% (354/1273) |

Chance is 25%. Medical training did not move 0-shot. 5-shot MMLU went down.
MedQA 5-shot is the only gain. 5-shot MedQA is hurt by `MAX_LEN = 2048` in
`scripts/isambard/student_mmlu.py`; long vignettes are cut off.

### 1.7B Python next-token — done

- Final job 7660092. Finished 23 Oct 2026 18:30, last leg 2 h 46 min,
  resumed from step 28000.
- Checkpoint `runs/pilot_1.7b_ntp/step_0030517.pt` (the log reached step
  30500 and the job exited 0; confirm the file before eval).
- Step 30500 loss 1.18.
- HumanEval has not been run.
- Yaml `config/pilot_1.7b.yaml` still says `resume: .../step_0028000.pt`.
  Submitting it again reloads step 28000 and retrains the last part.

### 1.7B Python distillation — done

- Job 7641700. Finished 23 Oct 2026 08:48, 10 h 43 min from step 21000.
  It survived the node failure that hit the other jobs.
- Checkpoint `runs/pilot_1.7b_kd/step_0030517.pt`.
- Step 30500: KL 0.40, logged hard-token loss 1.04.
- HumanEval job 7660233: **19/164 = 11.6%** (`pass_at_1` 0.1159).
  File `logs/python_kd_1.7b_humaneval.json`.
- Yaml still says `resume: .../step_0021000.pt`. Do not resubmit it.

### 1.7B Apollo next-token — done

- Final job 7660093. Finished 23 Oct 2026 18:33.
- Checkpoint `runs/pilot_1.7b_medical_apollo_ntp/step_0030517.pt`
  (same caveat: log reached step 30500, job exited 0).
- Step 30500 loss 2.50.
- Medical eval has not been run.
- Yaml resume is still step 28000.

### 0.6B Apollo distillation — done

- Final job 7660094. Finished 24 Oct 2026 02:50, 11 h 5 min from step 13000.
- Checkpoint `runs/pilot_0.6b_medical_apollo_kd/step_0030517.pt`.
- Step 30500: KL 0.64, logged hard-token loss 2.45 (Apollo next-token ended
  at 2.60).
- Evals: `logs/apollo_kd_0.6b_mmlu.json` (letter scoring, 27 Oct) and
  `logs/apollo_kd_0.6b_mmlu_cloze.json` (letter plus cloze, 27 Oct).

### 1.7B Apollo distillation — done

- Resumed from step 28000 on 27 Oct 2026 and finished at step 30517 in one
  short job. Checkpoint `runs/pilot_1.7b_medical_apollo_kd/step_0030517.pt`
  (23.02 GB, same size as its neighbours).
- Eval: `logs/apollo_kd_1.7b_mmlu.json`.

## Evals — all done

Every run has its exam file: `logs/python_*_humaneval.json` for the four
Python students and `logs/apollo_*_mmlu.json` for the four Apollo students.
Diagnostics per checkpoint (held-out CE and KL against the teacher, letter
calibration, cloze scores, greedy samples) are in `logs/diag/pilot_*.json`;
the per-shard forgetting sweeps are `logs/diag/forget_*.json`. The full grid
and the interpretation are in `PILOT_DIAGNOSIS.md`.

Scripts accept `CHECKPOINT` and `OUT`; defaults still point at the 0.6B
Python next-token files, so always set both.

Medical MMLU is five subjects, 945 questions; MedQA-USMLE is 1273 four-way
questions. `scripts/student_mmlu.py` now reports two scores. The letter
score is which of ` A`–` D` is most likely after `Answer:` (0 and 5 shots).
It is a letter-frequency bias for these students: every pilot student is at
or below chance and most pick A nearly always. The cloze score ranks the four
option texts after the question by summed and by per-token log probability
(0-shot only, keys `mmlu.cloze` and `medqa.cloze`). The per-token variant is
the one that moves: Apollo students 0.29–0.32 on MMLU and 0.28–0.31 on
MedQA against 0.25 for Python students, ordered distilled > next-token and
1.7B > 0.6B. Report cloze for from-scratch students; letter scoring will
only mean something with far more tokens.

HumanEval is greedy pass@1 on 164 tasks: `scripts/student_humaneval.slurm`.

## 27 Oct 2026: shuffling fix

The one real defect found: `pack_shards.py` wrote the packs in file order
and the trainer read them shard by shard. The Apollo corpus is three
sub-corpora in sequence and the Python set is sorted by educational score,
so each run saw one slice of its corpus after another and measurably forgot
the early slices (loss on Apollo shard 0 rose from 2.61 at step 5000 to
3.87 at step 16000 while the run was on shards 8–16). The teacher logits are
position-aligned with the existing packs and took about four days of 2-GPU
time per domain, so the packs were not rewritten. Instead:

- `src/train_kd.py`: `training.shuffle: true` draws sequences in a seeded
  random order (`seed` plus epoch) across all shards and sources; the
  logits stay aligned because sequences are permuted, not repacked. A
  config may now list several packs under `data:` (each with `dir`,
  optional `logit_dir`, optional `shards` cap) instead of `data_dir`. The
  cursor state is now `position`/`epoch`/`shuffle_seed`; old `shard`/`index`
  states load unchanged. Without `shuffle` the order is identical to the
  pilots. Resuming with a different `shuffle` setting than the checkpoint
  is refused.
- `scripts/pack_shards.py`: `--shuffle-seed` writes documents in random
  order (two passes over byte offsets) and rows may be JSON lists of
  strings, which is the Apollo question-answer format.
- `data/packed/medical_apollo_qa`: the Apollo exam-style question-answer
  file (417,236 rows, never used before) packed with shuffle seed 42:
  33,346 sequences, shards `[8192, 8192, 8192, 8192, 578]`, job 7766494.
  It has **no teacher logits yet**.
- Configs `config/pilot_0.6b_medical_apollo_{ntp,kd}_shuffled.yaml`: the
  0.6B Apollo pair with `shuffle: true` over the 30 logit-covered Apollo
  shards plus the QA pack (about 12% of the pool), same budget and schedule.
  Output dirs end in `_shuffled`.
- Checks run: a synthetic cursor test (legacy order, full coverage of one
  epoch, resume reproduces the next batch, shard cap); a 60-step real-data
  distillation smoke with random reads over all 30 logit shards
  (`runs/smoke_0.6b_kd_shuffled`, job 7766507, normal step rate, KL 10.8 to
  5.8); a cloze eval reproducing the diagnostic numbers exactly.

Submitted 27 Oct 2026 about 19:10 UTC, all with the 24 h limit:

1. Job 7771849 `apollo_qa_logits`: `scripts/precompute_logits.slurm` with
   `SHARDS=$S/data/packed/medical_apollo_qa` and
   `LOGIT_DIR=$S/data/logits/medical_apollo_qa`. Five shards, about 3 h
   each on 2 GPUs. Log `logs/apollo_qa_logits_7771849.log`. If it times
   out, resubmit the same command: the script skips finished shards.
2. Job 7771850 `apollo_ntp_0.6b_shuf`: `pilot_0.6b_medical_apollo_ntp_shuffled.yaml`,
   about 19 h on one GPU. Log `logs/pilot_0.6b_medical_apollo_ntp_shuffled_7771850.log`,
   checkpoints `runs/pilot_0.6b_medical_apollo_ntp_shuffled/`.
3. Job 7771851 `apollo_kd_0.6b_shuf`: `pilot_0.6b_medical_apollo_kd_shuffled.yaml`,
   held by `--dependency=afterok:7771849`, about 19 h once it starts.
   If the logits job fails, Slurm leaves this job pending forever with
   reason `DependencyNeverSatisfied`; cancel it and resubmit after the
   logits exist.
4. Done. Both finished (next-token 28 Oct 14:06 UTC, distilled 29 Oct
   03:59 UTC), evals in `logs/apollo_{ntp,kd}_0.6b_shuffled_mmlu.json`,
   per-shard loss sweeps in `logs/diag/forget_pilot_0.6b_medical_apollo_{ntp,kd}_shuffled.json`.
   Result (details in `PILOT_DIAGNOSIS.md`): forgetting gone in both; loss
   0.32 / 0.18 nats lower on average (next-token / distilled), 0.25 / 0.10
   lower on unseen shard 31; answer-text exam scores up 0 to 1.5 for
   next-token and 1.3 to 3.6 for distilled (MMLU 31.1, MedQA 30.6 at
   0-shot, 32.8 / 32.6 at 3-shot). The 0.6B shuffled distilled model matches
   the unshuffled 1.7B distilled one. Letter scoring still at chance.
5. Done on 30 Oct 2026, see the next section: QA pack decontaminated and
   repacked, 1.7B Apollo pair resubmitted shuffled. The Python pairs were
   not rerun (decision 30 Oct: HumanEval gain expected to be within noise).

The two 0.6B shuffled configs have no `resume:` key and still point at the
contaminated pack `medical_apollo_qa`; they are finished and are kept as
they ran. New runs use `RESUME_LATEST=1` (below) instead of a `resume:` key.

## 30 Oct 2026: decontamination and the 1.7B shuffled pair

- `scripts/decontaminate_qa.py` drops QA rows whose question stem (first 80
  normalised characters) equals a MedQA-USMLE test or medical MMLU test
  question. Job 7867538 dropped 30 of 417,236 rows, covering 26 distinct
  test prefixes (21 MedQA, 5 MMLU; the earlier count of 8 MMLU counted
  duplicated questions across subjects separately). Report
  `data/raw/apollo/qa/decontamination.json`, clean file
  `merged_apollo_en_qa_clean.jsonl`, pack `data/packed/medical_apollo_qa_clean`
  (33,344 seqs, shards 8192×4 + 576, shuffle seed 42).
- Teacher logits for the clean pack: job 7867701 →
  `data/logits/medical_apollo_qa_clean`, about 16 h.
- `src/train_kd.py --resume-latest` (set `RESUME_LATEST=1` for
  `train_pilot.slurm`): when the yaml has no `resume:` key, continue from
  the newest *loadable* `step_*.pt` in `output_dir`, skipping a file cut
  off by a wall-clock kill, or start fresh if the directory is empty.
  Verified by a unit test and a two-job smoke (jobs 7867534/7867536: junk
  `step_0000045.pt` skipped, resumed at step 30, finished at 60). This also
  makes a `NODE_FAIL` requeue safe for these runs: it resumes instead of
  restarting from 0.
- 1.7B Apollo shuffled runs, configs
  `config/pilot_1.7b_medical_apollo_{ntp,kd}_shuffled.yaml` (Apollo 30
  shards + clean QA pack, `shuffle: true`, same schedule, about 34 h at 15
  steps/min), each submitted as two 24 h jobs, the second
  `--dependency=afterany` on the first with `RESUME_LATEST=1`:
  - next-token: 7867702 → 7867703, `runs/pilot_1.7b_medical_apollo_ntp_shuffled`,
    log `logs/apollo_ntp_1.7b_shuffled_<job>.log`.
  - distilled: 7867704 (afterok logits 7867701) → 7867705,
    `runs/pilot_1.7b_medical_apollo_kd_shuffled`.
  - Evals chained `afterok` on the second job of each: `student_mmlu`
    7867706 / 7867708 → `logs/apollo_{ntp,kd}_1.7b_shuffled_mmlu.json`;
    `diag_forgetting` 7867707 / 7867709 →
    `logs/diag/forget_pilot_1.7b_medical_apollo_{ntp,kd}_shuffled.json`.
  - If a second job times out before step 30517 (run slower than 48 h),
    the evals go `DependencyNeverSatisfied`: cancel them, submit a third
    training job with `RESUME_LATEST=1`, and resubmit the evals after it.
  - Each 1.7B run keeps 30 checkpoints of 23 GB (690 GB); the project has
    about 195 TB free.
- Done 1 Nov 2026. All nine jobs completed; both runs took 24 h + about
  11 h (resumed from step 21000 by `--resume-latest`, as designed).
  Checkpoints `runs/pilot_1.7b_medical_apollo_{ntp,kd}_shuffled/step_0030517.pt`,
  evals `logs/apollo_{ntp,kd}_1.7b_shuffled_mmlu.json`, sweeps
  `logs/diag/forget_pilot_1.7b_medical_apollo_{ntp,kd}_shuffled.json`.
  Result (section "1 Nov 2026" in `PILOT_DIAGNOSIS.md`): no forgetting;
  mean loss 2.19 / 2.11 (next-token / distilled), unseen shard 2.86 / 2.55,
  about 0.10 nats below the shuffled 0.6B pair. Exams did not move: 1.7B
  distilled 31.0 / 32.8 MMLU and 31.2 / 31.8 MedQA (answer-text 0 / 3-shot),
  the same as the 0.6B shuffled distilled model. All Apollo students sit at
  29–33 on the exams; loss is the metric that separates them.
- Nothing is queued. Possible next steps, none started: rerun the Python
  pairs shuffled (HumanEval, no new logits needed; one 24 h job per 0.6B
  run, two per 1.7B run with `RESUME_LATEST=1`); or decide the token
  budget for the real grid (section 4 of `PILOT_DIAGNOSIS.md`).

## Measured picture, so far

At 1B tokens, 0.6B distillation did not beat next-token training on
HumanEval (6.7% vs 6.1%). The 1.7B distilled Python model did: 11.6%.
Apollo next-token training at 0.6B did not beat a Python-only model on
medical MMLU and was flat on MedQA 0-shot. These students are trained from
scratch on 1B tokens, about 1.4 tokens per parameter at 0.6B and about 0.5
at 1.7B. Do not compare them with published Qwen 0.6B or 1.7B numbers.
Those models saw orders of magnitude more text. See `PILOT_DIAGNOSIS.md`
for the full grid, the forgetting measurements and the scale anchors from
the literature.

## Environments and files

- Train: `/projects/u6wc/savants/venv-train`, torch 2.10.0+cu126, cuda/12.6.
- Logits: `/projects/u6wc/savants/venv-vllm126`.
- Compiler for vLLM JIT: `/projects/u6wc/savants/gcc13`.
- Logs: `/projects/u6wc/savants/logs/<jobname>_<jobid>.log`.
- Trainer source of truth to edit: `src/train_kd.py`.
- Cluster trainer: `/projects/u6wc/savants/src/train_kd.py`.

## 2 Nov 2026: scaling fit and report

Nothing is queued or running. Jakub (2 Nov) capped extra spend at about
1,000–1,500 GPU-hours, one domain, and asked for a scaling-law fit and a 1–2
page report before any larger run. Both done without new jobs:

- `reports/PILOT_REPORT.md`: the report (method, data, training, results,
  scaling fit, options).
- `scripts/isambard/fit_scaling.py` + `reports/pilot_results/curves/*.csv`
  + `reports/pilot_results/scaling_fit.json`: the fit (beta 0.40 passes the
  half-fit check; 1.7B KD predicted 2.36 on shard 31 at 9B tokens vs 2.55 now).
- `reports/pilot_results/teacher_ce_shard29_unseen.json`: teacher CE bracket
  (1.74 lower bound, ~1.8–1.9) on the same unseen rows where the 1.7B KD
  student scores 2.24. Computed on the login node from the stored logits.

Proposed next step (not approved yet): continue the 0.6B and 1.7B distilled
shuffled students for two more epochs over the existing logit-covered packs
(3.7B tokens total, about 170 GPU-h, no new logits) with shard-31 loss and
answer-text exams every 0.5B tokens; decide on the 9B PubMed run from that
curve. The trainer's cursor supports epochs (`epoch` in the cursor state), so
this is a config with `steps: 91551` resumed from `step_0030517.pt`; note the
LR schedule would restart its decay phase unless `decay_start_step` is moved.

## 4 Nov 2026: 3B-token continuation (Jakub approved ~170 GPU-h)

Scaling check: continue both shuffled distilled students from their final
checkpoint (step 30517, annealed) to step 91551 = 3.0B tokens on the same
pool, with exams and the shard-31 sweep at steps 45000/60000/75000/91551.

- Trainer: `training.rewarmup_from_step` + `rewarmup_steps` ramp the LR
  from the floor back to the schedule (needed because the checkpoint ended
  at 6e-5). `--resume-latest` now takes precedence over a config `resume:`
  path when output_dir already holds a checkpoint, so chained jobs continue
  their own run.
- Configs `config/pilot_{0.6b,1.7b}_medical_apollo_kd_shuffled_3b.yaml`
  (resume from the `_shuffled` step_0030517.pt, steps 91551, decay_start
  82396, save_every 5000, output `runs/..._shuffled_3b`). Smoke config
  `config/smoke_0.6b_kd_3b.yaml` (40 steps, save_every 20) to verify the
  resume, the re-warmup and the resume-latest precedence before the chain.
- `scripts/isambard/submit_3b.sh SIZE JOBS` (cluster copy `scripts/submit_3b.sh`)
  submits the chain plus evals: `logs/apollo_kd_SIZE_3b_stepN_mmlu.json`,
  `logs/diag/forget_pilot_SIZE_medical_apollo_kd_shuffled_3b.json`.
- Expected: 0.6B two 24 h jobs (~39 h), 1.7B three (~68 h), evals ~15 GPU-h;
  total ~120 GPU-h. Prediction to check (beta 0.40 fit): 1.7B held-out shard
  31 = 2.44 at 3B tokens (2.55 at 1B).
- Disk: 1.7B checkpoints are 23 GB each; 13 new ones = 300 GB. Check quota
  before submitting (`df -h /projects/u6wc`).

Submitted 4 Nov 2026 08:35 UTC (all pending until the cooling shutdown ends
at 17:00 UTC): smoke 8064644 (`logs/smoke_kd_3b_8064644.log`, gates both
chains with afterok). 0.6B: train 8064645 → 8064646, exams 8064647–8064650,
sweep 8064651. 1.7B: train 8064652 → 8064653 → 8064654, exams
8064655–8064658, sweep 8064659. Logs `logs/apollo_kd_{0.6b,1.7b}_3b_<job>.log`.
If the smoke fails, everything downstream is cancelled by Slurm
(DependencyNeverSatisfied); fix, rerun the smoke, and resubmit with
`AFTER=<new smoke> bash scripts/submit_3b.sh SIZE JOBS`.
5 Nov 2026 12:00 UTC: smoke passed; both chains running since 4 Nov 18:45 UTC
(0.6B job 8064645 at step 57k, 1.7B job 8064652 at 46k). Early exam evals on
the step 45000 checkpoints submitted as 8090711 (0.6B) / 8090712 (1.7B),
outputs `logs/apollo_kd_{0.6b,1.7b}_3b_step45000_mmlu.json`; the queued
duplicates 8064647 / 8064655 were cancelled.
6 Nov 2026 12:30 UTC: 0.6B chain finished (8064646 completed after 17 h);
evals 8064648–8064651 done, results copied to `reports/pilot_results/`
(3B final: MMLU 33.9 / 35.3, MedQA 30.3 / 31.9, shard 31 loss 2.535).
1.7B on job 8064653 at step 65.6k; early eval of its step 60000 submitted
as 8131323 (queued duplicate 8064656 cancelled). Remaining queue: 8064654
(third 1.7B job), 8064657/8064658 exams, 8064659 sweep.
1.7B mid-run exams (constant LR, not annealed): step 45000 MMLU 32.3 / 34.4,
MedQA 31.0 / 32.7; step 60000 (8131323) MMLU 32.4 / 32.7, MedQA 30.6 / 29.3.
8 Nov 2026 14:00 UTC: 1.7B job 3 (8064654) timed out at step 91320, 231
steps short; the afterok evals 8064657/8064658/8064659 were cancelled by
Slurm. Resubmitted: finishing job 8200119 (4 h limit, resumes from
step_0090000.pt), exam on step 75000 8200120 (no dependency), final exam
8200121 and shard sweep 8200122 afterok on 8200119. Lesson: at 14.7
steps/min a 24 h job covers 21,100 steps, and each resume loses up to 5,000
steps of progress with save_every 5000, so a 61k-step continuation needs
four jobs, not three.
8 Nov 2026 17:30 UTC: 1.7B finished (8200119), evals 8200120–8200122 done.
Final 3B: MMLU 33.1 / 36.7, MedQA 32.0 / 32.5, shard 31 loss 2.404 (2.44
predicted). Both continuations complete; results, refit
(`reports/pilot_results/scaling_fit_3b.json`) and notes committed;
PILOT_REPORT.md section 8 added. Nothing queued. Continuation cost about
113 GPU-h. Loss curves of the continuation runs: `logs/apollo_kd_{0.6b,1.7b}_3b_*.log`
(CSV copies in `reports/pilot_results/curves/*_3b.csv` if the copy went through).

## 9 Nov 2026: plan for the remaining budget (Jakub: 1,500–2,000 GPU-h max, two branches)

Jakub (9 Nov): go further on scaling, NTP is low priority (only show robustly
that it stays behind distillation), then focus on distilled/mixed training; a
"minimal general knowledge" corpus may be what the teacher had and we lack.
Details left to Marek. Everything below is prepared; the long jobs wait for
Marek's go-ahead. Deadline logic: 1.7B trains 1B tokens in ~35 h, logits cost
~190 GPU-h per 1B tokens at 2.8k tok/s on two GPUs, so the binding constraints
are wall clock (project ends 24 Nov) and logits, not GPU-hours.

Facts found today:
- The `data/packed/medical` pack (287 shards, 9.6B tokens, no logits) is in
  file order: shards 0–32 are the Apollo text (same documents as
  `medical_apollo`, including the held-out shard 31), shards 33–286 are PubMed
  abstracts in date order. PubMed duplicates of Apollo text were removed when
  it was built.
- Both NTP shuffled runs have held-out points at the same six checkpoints as
  the distilled runs, so an NTP-only law on 0.16–1B exists already
  (`scaling_fit.json`): gap NTP−KD is flat, 0.29–0.32 nats (0.6B) and
  0.30–0.36 (1.7B); NTP needs 2.5–4x the tokens for the same held-out loss.

Code (commit 05a1979, synced): shard lists in `data[].shards`,
`training.reset_cursor`, `--shard-numbers`/`SHARD_NUMBERS` in the logit
script, `--max-shards` in `pack_shards.py`, `scripts/submit_chain.sh`,
`scripts/submit_logits.sh`, `scripts/download_cosmopedia.sh`,
`scripts/pack_cosmopedia.slurm`.

Submitted 9 Nov ~23:00 UTC (short jobs): smoke 8246947
(`config/smoke_1.7b_kd_shardlist.yaml`, 40 steps from the 1.7B 3B checkpoint
over shards [0,15,29], tests the list path and reset_cursor); PubMed held-out
baseline on the 3B checkpoints 8246950 (0.6B) / 8246952 (1.7B), medical pack
shards 35 100 200 285 → `logs/diag/forget_pilot_SIZE_medical_apollo_kd_shuffled_3b_medical.json`;
Cosmopedia pack 8247068 (two parquet files downloaded to
`data/raw/cosmopedia`, pack → `data/packed/cosmopedia`, 9 shards = 0.3B tokens).

Planned chains (need go-ahead; commands run from /projects/u6wc/savants):

1. Branch 1a, NTP twin to 3B (≈70 GPU-h, 4 × 24 h):
   `NAME=apollo_ntp_1.7b_3b CONFIG=config/pilot_1.7b_medical_apollo_ntp_shuffled_3b.yaml RUN_DIR=runs/pilot_1.7b_medical_apollo_ntp_shuffled_3b JOBS=4 EVAL_STEPS="60000 91551" SWEEPS="data/packed/medical_apollo:0 4 8 12 16 20 24 28 31" bash scripts/submit_chain.sh`
2. Branch 1b, teacher logits for 90 PubMed shards (≈610 GPU-h, 4 chains × 4 × 24 h on 2 GPUs, ~3.3 days):
   `PACK=data/packed/medical LOGIT_DIR=data/logits/medical SHARDS=<list in config/pilot_1.7b_medical_pubmed_kd_6b.yaml> CHAINS=4 JOBS=4 bash scripts/submit_logits.sh`
   (shard list: 33,36,...,286, 90 numbers; logits land in `data/logits/medical/shard-000NN.*`, ~1.1 TB).
3. Branch 1b, distilled 3B→6B on the new tokens after the logits (0.6B ≈60 GPU-h 3 jobs + 1 finishing; 1.7B ≈105 GPU-h 5 + 1 finishing; must start by ~13 Nov):
   `NAME=pubmed_kd_SIZE_6b CONFIG=config/pilot_SIZE_medical_pubmed_kd_6b.yaml RUN_DIR=runs/pilot_SIZE_medical_pubmed_kd_6b JOBS=N EVAL_STEPS="120000 150000 183102" SWEEPS="data/packed/medical_apollo:0 4 8 12 16 20 24 28 31;data/packed/medical:35 100 200 285" bash scripts/submit_chain.sh`
   Gate the first job on the last logit jobs (`logs/logits_medical_last_jobs.txt`) or submit when all 90 `.done` files exist.
4. Branch 2a, Cosmopedia logits (9 shards ≈60 GPU-h, 1 chain × 2 jobs) after pack 8247068:
   `PACK=data/packed/cosmopedia LOGIT_DIR=data/logits/cosmopedia SHARDS=0,1,2,3,4,5,6,7,8 CHAINS=1 JOBS=2 bash scripts/submit_logits.sh`
   then mix vs control, both sizes (0.6B 20 GPU-h each, 1.7B 35 each; 1 and 2 jobs):
   `NAME=cosmo_mix_SIZE CONFIG=config/pilot_SIZE_cosmo_mix_kd.yaml RUN_DIR=runs/pilot_SIZE_cosmo_mix_kd JOBS=N EVAL_STEPS="122068" SWEEPS="data/packed/medical_apollo:0 4 8 12 16 20 24 28 31" bash scripts/submit_chain.sh`
   `NAME=apollo_e4_SIZE CONFIG=config/pilot_SIZE_apollo_e4_kd.yaml RUN_DIR=runs/pilot_SIZE_apollo_e4_kd JOBS=N EVAL_STEPS="122068" SWEEPS=... bash scripts/submit_chain.sh`
   (control needs no new logits, can start right away).
5. Branch 2b, teacher-generated medical rewrites with logprobs captured at
   generation time: pipeline not written yet; only if time remains.

Budget: ≈70 + 610 + 165 + 60 + 110 + evals ≈ 1,050 GPU-h.
