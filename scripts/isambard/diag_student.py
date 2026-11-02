#!/usr/bin/env python3
"""Diagnostics for one from-scratch student checkpoint.

Everything here is read-only on the checkpoint and writes a single JSON:

- Held-out loss and forward KL to the teacher on rows of shard 29 that lie
  past the training cursor, for both domains (in-domain and cross-domain).
- Exam letter log-probabilities for medical MMLU and MedQA, 0 and 5 shot,
  with and without a leading EOS token, plus letter-pick histograms and a
  content-free calibration (subtract each letter's mean log-probability).
- Cloze scoring: mean and summed log-probability of each option text given
  the question, 0-shot, for MedQA and medical MMLU.
- A few greedy continuations for a qualitative look.
"""

import argparse
import gc
import importlib.util
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

S = "/projects/u6wc/savants"
TOKENIZER = f"{S}/models/Qwen3.5-35B-A3B-Base"
EOS = 248044
LETTERS = "ABCD"
DOMAINS = {
    "python": (f"{S}/data/packed/python", f"{S}/data/logits/python"),
    "apollo": (f"{S}/data/packed/medical_apollo", f"{S}/data/logits/medical_apollo"),
}
# Training consumed 30517 * 8 = 244136 sequences: shard 29 up to index 6568.
HELDOUT_SHARD = 29
HELDOUT_ROW0 = 7168
HELDOUT_ROWS = 128
MAX_LEN = 2048
BATCH = 8
SAMPLE_PROMPTS = [
    "The patient presented with chest pain and",
    "Question: What is the function of the mitral valve?\nAnswer:",
    "def fibonacci(n):\n",
    "The following are multiple choice questions (with answers) about anatomy.\n\nWhich bone is the longest in the human body?\nA. Tibia\nB. Femur\nC. Humerus\nD. Fibula\nAnswer:",
]


def load_module(name, relative):
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in relative:
        path = os.path.abspath(os.path.join(here, *rel))
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit(f"{name} not found")


trainer = load_module("train_kd", (("..", "src", "train_kd.py"), ("..", "..", "src", "train_kd.py")))
exam = load_module("student_mmlu", (("student_mmlu.py",), ("..", "scripts", "student_mmlu.py")))


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = trainer.build_student(trainer.config_for_checkpoint(state), for_training=False)
    model.load_state_dict(state["model"])
    info = {"model_size": state.get("model_size", "0.6b"), "objective": state.get("objective"),
            "step": state.get("step"), "tokens": state.get("tokens")}
    del state
    gc.collect()
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    return model, info


def hidden_states(model, input_ids, attention_mask=None):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.last_hidden_state


def heldout(model, weight, domain):
    packed, logits = DOMAINS[domain]
    with open(os.path.join(packed, "meta.json")) as handle:
        meta = json.load(handle)
    seq_len = meta["seq_len"]
    count = meta["shards"][HELDOUT_SHARD]
    tok = np.memmap(os.path.join(packed, f"shard-{HELDOUT_SHARD:05d}.bin"), dtype=np.uint32, mode="r", shape=(count, seq_len))
    ids = np.load(os.path.join(logits, f"shard-{HELDOUT_SHARD:05d}.ids.npy"), mmap_mode="r")
    lp = np.load(os.path.join(logits, f"shard-{HELDOUT_SHARD:05d}.lp.npy"), mmap_mode="r")
    ce_sum = kl_sum = teacher_sum = 0.0
    positions = covered = 0
    top1_match = 0
    for start in range(HELDOUT_ROW0, HELDOUT_ROW0 + HELDOUT_ROWS, 4):
        stop = start + 4
        batch = torch.from_numpy(np.asarray(tok[start:stop]).astype(np.int64)).cuda()
        tid = torch.from_numpy(np.asarray(ids[start:stop, 1:]).astype(np.int64)).cuda()
        tlp = torch.from_numpy(np.asarray(lp[start:stop, 1:]).astype(np.float32)).cuda()
        hidden = hidden_states(model, batch)[:, :-1, :].float()
        labels = batch[:, 1:]
        with torch.inference_mode():
            ce = trainer.cross_entropy(hidden, weight, labels, 1024)
            kl = trainer.topk_forward_kl(hidden, weight, tid, tlp, 1024)
            # student top-1 agreement with the true token
            flat = hidden.reshape(-1, hidden.shape[-1])
            pred = torch.cat([F.linear(flat[i:i + 2048], weight).argmax(-1) for i in range(0, flat.shape[0], 2048)])
            top1_match += (pred == labels.reshape(-1)).sum().item()
        n = labels.numel()
        ce_sum += ce.item() * n
        kl_sum += kl.item() * n
        positions += n
        match = tid == labels.unsqueeze(-1)
        have = match.any(-1)
        teacher_sum += -(torch.where(match, tlp, torch.zeros_like(tlp)).sum(-1)[have]).sum().item()
        covered += have.sum().item()
    return {
        "tokens": positions,
        "ce": ce_sum / positions,
        "ppl": math.exp(ce_sum / positions),
        "kl_to_teacher": kl_sum / positions,
        "student_top1_acc": top1_match / positions,
        "teacher_ce_on_covered": teacher_sum / max(1, covered),
        "teacher_top64_cover": covered / positions,
    }


def last_logprobs(model, weight, tokenizer, prompts, choice_ids, prepend_eos):
    """(n, 4) log-probabilities of the letter tokens at the last position; NaN when the prompt is too long."""
    out = np.full((len(prompts), 4), np.nan, dtype=np.float64)
    encoded = []
    kept = []
    for index, text in enumerate(prompts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if prepend_eos:
            ids = [EOS] + ids
        if len(ids) + 1 > MAX_LEN:
            continue
        encoded.append(ids)
        kept.append(index)
    choice = torch.tensor(choice_ids, device="cuda")
    for start in range(0, len(encoded), BATCH):
        chunk = encoded[start:start + BATCH]
        width = max(len(ids) for ids in chunk)
        input_ids = torch.tensor([[EOS] * (width - len(ids)) + ids for ids in chunk], device="cuda")
        mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in chunk], device="cuda")
        last = hidden_states(model, input_ids, mask)[:, -1, :].float()
        with torch.inference_mode():
            logprobs = F.log_softmax(F.linear(last, weight), dim=-1).index_select(-1, choice)
        out[kept[start:start + len(chunk)]] = logprobs.cpu().numpy()
    return out


def summarize_letters(values, gold):
    """Accuracy raw and calibrated, pick histogram, mean margin."""
    ok = ~np.isnan(values[:, 0])
    v = values[ok]
    g = np.array(gold)[ok]
    picks = v.argmax(-1)
    raw = float((picks == g).mean()) if len(g) else 0.0
    calibrated = v - v.mean(0, keepdims=True)
    cal_picks = calibrated.argmax(-1)
    cal = float((cal_picks == g).mean()) if len(g) else 0.0
    hist = np.bincount(picks, minlength=4).tolist()
    cal_hist = np.bincount(cal_picks, minlength=4).tolist()
    gold_hist = np.bincount(g, minlength=4).tolist()
    # probability mass on the four letters together (how exam-shaped the model is)
    mass = float(np.exp(v).sum(-1).mean()) if len(g) else 0.0
    return {
        "n": int(len(gold)), "answered": int(ok.sum()), "accuracy": round(raw, 4),
        "calibrated_accuracy": round(cal, 4), "pick_hist_ABCD": hist,
        "calibrated_pick_hist_ABCD": cal_hist, "gold_hist_ABCD": gold_hist,
        "mean_letter_mass": round(mass, 4),
        "mean_letter_logprob_ABCD": [round(float(x), 3) for x in v.mean(0)] if len(g) else None,
    }


def exam_sets(mmlu_dir):
    """Returns dict name -> (prompts_by_shot, gold indexes, (question, options) for cloze)."""
    sets = {}
    rows_all, gold_all, cloze_all = [], [], []
    prompts = {0: [], 5: []}
    for key, label in exam.SUBJECTS.items():
        dev = exam.load_jsonl(os.path.join(mmlu_dir, f"{key}_dev.jsonl"))
        test = exam.load_jsonl(os.path.join(mmlu_dir, f"{key}_test.jsonl"))
        shots = [(r["question"], r["choices"], LETTERS[r["answer"]]) for r in dev]
        for n_shot in (0, 5):
            prompts[n_shot].extend(exam.exam_prompt(label, shots[:n_shot], r["question"], r["choices"]) for r in test)
        gold_all.extend(r["answer"] for r in test)
        cloze_all.extend((r["question"], r["choices"]) for r in test)
    sets["mmlu"] = (prompts, gold_all, cloze_all)

    test = exam.load_jsonl(exam.MEDQA_TEST)
    train = exam.load_jsonl(exam.MEDQA_TRAIN, limit=5)
    shots = [(r["question"], [r["options"][l] for l in LETTERS], r["answer_idx"]) for r in train]
    prompts = {0: [], 5: []}
    for n_shot in (0, 5):
        prompts[n_shot] = [exam.exam_prompt("medicine", shots[:n_shot], r["question"], [r["options"][l] for l in LETTERS]) for r in test]
    gold = [LETTERS.index(r["answer_idx"]) for r in test]
    cloze = [(r["question"], [r["options"][l] for l in LETTERS]) for r in test]
    sets["medqa"] = (prompts, gold, cloze)
    return sets


def cloze_scores(model, weight, tokenizer, items):
    """Per item, per option: (sum logprob, mean logprob) of the option text after 'question\\n'."""
    seqs = []
    owners = []
    for index, (question, options) in enumerate(items):
        q = tokenizer.encode(question.strip() + "\n", add_special_tokens=False)
        for opt in options:
            o = tokenizer.encode(opt.strip(), add_special_tokens=False)
            if not o:
                o = [EOS]
            ids = ([EOS] + q + o)[-MAX_LEN:]
            seqs.append((ids, len(o)))
            owners.append(index)
    sums = np.zeros((len(items), 4))
    means = np.zeros((len(items), 4))
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]))
    slot = [0] * len(items)
    for start in range(0, len(order), 16):
        group = order[start:start + 16]
        width = max(len(seqs[k][0]) for k in group)
        input_ids = torch.tensor([[EOS] * (width - len(seqs[k][0])) + seqs[k][0] for k in group], device="cuda")
        mask = torch.tensor([[0] * (width - len(seqs[k][0])) + [1] * len(seqs[k][0]) for k in group], device="cuda")
        hidden = hidden_states(model, input_ids, mask)
        for row, k in enumerate(group):
            ids, n_opt = seqs[k]
            # positions predicting the option tokens: the n_opt hidden states before the last n_opt tokens
            h = hidden[row, width - n_opt - 1:width - 1, :].float()
            targets = input_ids[row, width - n_opt:width]
            with torch.inference_mode():
                lp = F.log_softmax(F.linear(h, weight), dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
            total = lp.sum().item()
            owner = owners[k]
            j = k % 4
            sums[owner, j] = total
            means[owner, j] = total / n_opt
    return sums, means


def cloze_summary(sums, means, gold):
    g = np.array(gold)
    return {
        "n": int(len(g)),
        "accuracy_sum_logprob": round(float((sums.argmax(-1) == g).mean()), 4),
        "accuracy_mean_logprob": round(float((means.argmax(-1) == g).mean()), 4),
        "pick_hist_sum": np.bincount(sums.argmax(-1), minlength=4).tolist(),
        "pick_hist_mean": np.bincount(means.argmax(-1), minlength=4).tolist(),
    }


def samples(model, tokenizer):
    out = []
    for text in SAMPLE_PROMPTS:
        ids = torch.tensor([[EOS] + tokenizer.encode(text, add_special_tokens=False)], device="cuda")
        with torch.inference_mode():
            gen = model.generate(ids, max_new_tokens=60, do_sample=False, eos_token_id=EOS, pad_token_id=EOS)
        out.append({"prompt": text, "continuation": tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mmlu-dir", default=exam.MMLU_DIR)
    parser.add_argument("--skip-cloze", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    choice_ids = exam.letter_ids(tokenizer)
    model, info = load_model(args.checkpoint)
    weight = model.lm_head.weight.float()
    result = {"checkpoint": args.checkpoint, **info}
    print("loaded", json.dumps(info), flush=True)

    result["heldout"] = {}
    for domain in DOMAINS:
        result["heldout"][domain] = heldout(model, weight, domain)
        print("heldout", domain, json.dumps(result["heldout"][domain]), flush=True)

    sets = exam_sets(args.mmlu_dir)
    result["exams"] = {}
    for name, (prompts, gold, _) in sets.items():
        for n_shot in (0, 5):
            for prepend in (False, True):
                values = last_logprobs(model, weight, tokenizer, prompts[n_shot], choice_ids, prepend)
                key = f"{name}_{n_shot}shot{'_eos' if prepend else ''}"
                result["exams"][key] = summarize_letters(values, gold)
                print(key, json.dumps(result["exams"][key]), flush=True)

    if not args.skip_cloze:
        result["cloze"] = {}
        for name, (_, gold, items) in sets.items():
            sums, means = cloze_scores(model, weight, tokenizer, items)
            result["cloze"][name] = cloze_summary(sums, means, gold)
            print("cloze", name, json.dumps(result["cloze"][name]), flush=True)

    result["samples"] = samples(model, tokenizer)
    for s in result["samples"]:
        print("SAMPLE", json.dumps(s), flush=True)

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
