# Pilot result files

Copied from `/projects/u6wc/savants/logs/` on Isambard-AI on 2 Nov 2026.
Interpretation is in `PILOT_DIAGNOSIS.md` (repo root); run bookkeeping is in
`AGENT_HANDOFF.md`.

- `apollo_{ntp,kd}_{0.6b,1.7b}_*mmlu*.json`: medical MMLU (945 questions, 5
  subjects) and MedQA-USMLE (1273) for the Apollo students. Keys `0-shot` and
  `5-shot` are letter scoring; `cloze` and `cloze_3-shot` are answer-text
  scoring (`accuracy_mean` is the per-token score reported in the write-up).
  `_shuffled` files are the reruns with sequence shuffling.
- `python_*_humaneval.json`: greedy pass@1 HumanEval for the Python students.
- `diag/pilot_*.json`: held-out cross-entropy and KL against the teacher,
  letter calibration and greedy samples per final checkpoint.
- `diag/forget_*.json`: loss on fixed rows of shards 0,4,...,28 and unseen
  shard 31 at six checkpoints (the forgetting sweeps).
- `qa_decontamination.json`: Apollo QA rows dropped for matching an exam
  test question (30 Oct 2026).
