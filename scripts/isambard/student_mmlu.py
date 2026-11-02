#!/usr/bin/env python3
"""Medical MMLU and MedQA for a from-scratch Qwen3 student checkpoint.

Same exam layout as the teacher run: a subject line, optional answered
examples, choices A–D, and the word Answer. The letter score is which of the
four letter tokens is most probable as the next token. The checkpoint is the
state dict written by src/train_kd.py.

Letter scoring is a letter-frequency bias for these small from-scratch
students (the October 2026 pilots picked A almost always), so the 0-shot
exams are also cloze scored: the question followed by each option's text,
ranked by the summed and by the per-token log probability of the option.
The per-token variant (lm-eval's acc_norm style) is the one that separates
the pilot students from chance.
"""

import argparse
import gc
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

CHECKPOINT = "/projects/u6wc/savants/runs/pilot_0.6b_ntp/step_0030517.pt"
TOKENIZER = "/projects/u6wc/savants/models/Qwen3.5-35B-A3B-Base"
MMLU_DIR = "/projects/u6wc/savants/data/raw/eval/mmlu_medical"
MEDQA_TEST = "/projects/u6wc/savants/data/raw/eval/GBaker__MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl"
MEDQA_TRAIN = "/projects/u6wc/savants/data/raw/eval/GBaker__MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl"
EOS = 248044
MAX_LEN = 2048
BATCH = 8
LETTERS = "ABCD"
CLOZE_SHOTS = 3
SUBJECTS = {
    "anatomy": "anatomy",
    "clinical_knowledge": "clinical knowledge",
    "college_medicine": "college medicine",
    "medical_genetics": "medical genetics",
    "professional_medicine": "professional medicine",
}


def train_kd():
    here = os.path.dirname(os.path.abspath(__file__))
    for relative in (("..", "src", "train_kd.py"), ("..", "..", "src", "train_kd.py")):
        path = os.path.abspath(os.path.join(here, *relative))
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("train_kd", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit("src/train_kd.py not found")


def load_jsonl(path, limit=None):
    rows = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def item_text(question, choices, answer=None):
    lines = [question.strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice.strip()}")
    text = "\n".join(lines) + "\nAnswer:"
    if answer is not None:
        text += " " + answer
    return text


def exam_prompt(subject, shots, question, choices):
    description = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    parts = [item_text(question_, choices_, answer) for question_, choices_, answer in shots]
    parts.append(item_text(question, choices))
    return description + "\n\n".join(parts)


def letter_ids(tokenizer):
    boundary = tokenizer.encode("Answer:", add_special_tokens=False)
    ids = []
    for letter in LETTERS:
        full = tokenizer.encode(f"Answer: {letter}", add_special_tokens=False)
        if full[:-1] != boundary or len(full) != len(boundary) + 1:
            raise SystemExit(f"letter {letter} is not a single token after Answer:")
        ids.append(full[-1])
    return ids


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    trainer = train_kd()
    model = trainer.build_student(trainer.config_for_checkpoint(state), for_training=False)
    model.load_state_dict(state["model"])
    size = state.get("model_size", "0.6b")
    del state
    gc.collect()
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    print(f"loaded student {size}", flush=True)
    return model


def predict(model, tokenizer, prompts, choices):
    """Return the chosen letter index for each prompt, or None if it does not fit."""
    encoded = []
    kept = []
    for index, text in enumerate(prompts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + 1 > MAX_LEN:
            continue
        encoded.append(ids)
        kept.append(index)
    picks = {}
    choice_ids = torch.tensor(choices, device="cuda")
    for start in range(0, len(encoded), BATCH):
        chunk = encoded[start:start + BATCH]
        width = max(len(ids) for ids in chunk)
        input_ids = []
        mask = []
        for ids in chunk:
            pad = width - len(ids)
            input_ids.append([EOS] * pad + ids)
            mask.append([0] * pad + [1] * len(ids))
        tokens = torch.tensor(input_ids, device="cuda")
        attention = torch.tensor(mask, device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=tokens, attention_mask=attention).logits[:, -1, :]
        chosen = torch.argmax(logits.index_select(-1, choice_ids), dim=-1)
        for offset, letter in enumerate(chosen.tolist()):
            picks[kept[start + offset]] = letter
        done = min(start + BATCH, len(encoded))
        print(f"scored {done}/{len(encoded)}", flush=True)
    return picks, len(prompts) - len(kept)


def shots_text(shots):
    """Answered examples for answer-text scoring: question, newline, the correct answer's text."""
    return "".join(f"{question.strip()}\n{answer.strip()}\n\n" for question, answer in shots)


def cloze_scores(model, tokenizer, items):
    """Per item and option: summed and per-token log probability of the option text after the question.

    Each item is (question, options) or (question, options, prefix); the prefix
    holds answered examples from ``shots_text`` and goes before the question.
    """
    weight = model.lm_head.weight.float()
    seqs, owners = [], []
    for index, item in enumerate(items):
        question, options = item[0], item[1]
        prefix = item[2] if len(item) > 2 else ""
        q = tokenizer.encode(prefix + question.strip() + "\n", add_special_tokens=False)
        for option in options:
            o = tokenizer.encode(option.strip(), add_special_tokens=False) or [EOS]
            ids = ([EOS] + q + o)[-MAX_LEN:]
            seqs.append((ids, len(o)))
            owners.append(index)
    sums = np.zeros((len(items), len(LETTERS)))
    means = np.zeros((len(items), len(LETTERS)))
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]))
    for start in range(0, len(order), BATCH * 2):
        group = order[start:start + BATCH * 2]
        width = max(len(seqs[k][0]) for k in group)
        input_ids = torch.tensor(
            [[EOS] * (width - len(seqs[k][0])) + seqs[k][0] for k in group], device="cuda",
        )
        mask = torch.tensor(
            [[0] * (width - len(seqs[k][0])) + [1] * len(seqs[k][0]) for k in group], device="cuda",
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
        for row, k in enumerate(group):
            n_opt = seqs[k][1]
            # The n_opt hidden states before the last n_opt tokens predict the option tokens.
            h = hidden[row, width - n_opt - 1:width - 1, :].float()
            targets = input_ids[row, width - n_opt:width]
            with torch.inference_mode():
                lp = F.log_softmax(F.linear(h, weight), dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
            total = lp.sum().item()
            sums[owners[k], k % len(LETTERS)] = total
            means[owners[k], k % len(LETTERS)] = total / n_opt
        done = min(start + BATCH * 2, len(order))
        if done % (BATCH * 20) == 0 or done == len(order):
            print(f"cloze {done}/{len(order)}", flush=True)
    return sums, means


def cloze_accuracy(sums, means, gold):
    g = np.asarray(gold)
    pick_sum = sums.argmax(-1)
    pick_mean = means.argmax(-1)
    return {
        "n": int(len(g)),
        "accuracy_sum": round(float((pick_sum == g).mean()), 4) if len(g) else 0.0,
        "accuracy_mean": round(float((pick_mean == g).mean()), 4) if len(g) else 0.0,
        "pick_hist_sum": np.bincount(pick_sum, minlength=len(LETTERS)).tolist(),
        "pick_hist_mean": np.bincount(pick_mean, minlength=len(LETTERS)).tolist(),
    }


def accuracy(rows, picks, gold):
    correct = answered = 0
    for index, row in enumerate(rows):
        letter = picks.get(index)
        if letter is None:
            continue
        answered += 1
        correct += int(LETTERS[letter] == gold(row))
    return {
        "n": len(rows),
        "answered": answered,
        "correct": correct,
        "accuracy": round(correct / answered, 4) if answered else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--mmlu-dir", default=MMLU_DIR)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    choices = letter_ids(tokenizer)
    model = load_model(args.checkpoint)
    print(f"checkpoint {args.checkpoint}", flush=True)

    summary = {"checkpoint": args.checkpoint, "mmlu": {}, "medqa": {}}
    grouped = {}
    flat_rows = []
    flat_subjects = []
    flat_prompts = {0: [], 5: []}
    for key, label in SUBJECTS.items():
        dev = load_jsonl(os.path.join(args.mmlu_dir, f"{key}_dev.jsonl"))
        test = load_jsonl(os.path.join(args.mmlu_dir, f"{key}_test.jsonl"))
        shots = [(row["question"], row["choices"], LETTERS[row["answer"]]) for row in dev]
        grouped[label] = (shots, test)
        for n_shot in (0, 5):
            used = shots[:n_shot]
            flat_prompts[n_shot].extend(
                exam_prompt(label, used, row["question"], row["choices"]) for row in test
            )
        flat_rows.extend(test)
        flat_subjects.extend([label] * len(test))

    for n_shot in (0, 5):
        print(f"mmlu {n_shot}-shot", flush=True)
        picks, skipped = predict(model, tokenizer, flat_prompts[n_shot], choices)
        overall = accuracy(flat_rows, picks, lambda row: LETTERS[row["answer"]])
        overall["skipped_length"] = skipped
        by_subject = {}
        for label in grouped:
            indexes = [i for i, subject in enumerate(flat_subjects) if subject == label]
            subset = [flat_rows[i] for i in indexes]
            remapped = {new: picks[old] for new, old in enumerate(indexes) if old in picks}
            by_subject[label] = accuracy(subset, remapped, lambda row: LETTERS[row["answer"]])
        summary["mmlu"][f"{n_shot}-shot"] = {"overall": overall, "subjects": by_subject}
        print(f"MMLU {n_shot}-shot", json.dumps(summary["mmlu"][f"{n_shot}-shot"]), flush=True)

    print("mmlu cloze", flush=True)
    sums, means = cloze_scores(model, tokenizer, [(row["question"], row["choices"]) for row in flat_rows])
    gold = [row["answer"] for row in flat_rows]
    cloze = {"overall": cloze_accuracy(sums, means, gold), "subjects": {}}
    for label in grouped:
        indexes = [i for i, subject in enumerate(flat_subjects) if subject == label]
        cloze["subjects"][label] = cloze_accuracy(sums[indexes], means[indexes], [gold[i] for i in indexes])
    summary["mmlu"]["cloze"] = cloze
    print("MMLU cloze", json.dumps(cloze), flush=True)

    print(f"mmlu cloze {CLOZE_SHOTS}-shot", flush=True)
    prefixes = {
        label: shots_text([(q, choices[LETTERS.index(a)]) for q, choices, a in shots[:CLOZE_SHOTS]])
        for label, (shots, _) in grouped.items()
    }
    sums, means = cloze_scores(
        model, tokenizer,
        [(row["question"], row["choices"], prefixes[subject]) for row, subject in zip(flat_rows, flat_subjects)],
    )
    cloze = {"overall": cloze_accuracy(sums, means, gold), "subjects": {}}
    for label in grouped:
        indexes = [i for i, subject in enumerate(flat_subjects) if subject == label]
        cloze["subjects"][label] = cloze_accuracy(sums[indexes], means[indexes], [gold[i] for i in indexes])
    summary["mmlu"][f"cloze_{CLOZE_SHOTS}-shot"] = cloze
    print(f"MMLU cloze {CLOZE_SHOTS}-shot", json.dumps(cloze), flush=True)

    test = load_jsonl(MEDQA_TEST)
    train = load_jsonl(MEDQA_TRAIN, limit=5)
    shots = [
        (row["question"], [row["options"][letter] for letter in LETTERS], row["answer_idx"])
        for row in train
    ]
    for n_shot in (0, 5):
        print(f"medqa {n_shot}-shot", flush=True)
        used = shots[:n_shot]
        prompts = [
            exam_prompt("medicine", used, row["question"], [row["options"][letter] for letter in LETTERS])
            for row in test
        ]
        picks, skipped = predict(model, tokenizer, prompts, choices)
        result = accuracy(test, picks, lambda row: row["answer_idx"])
        result["skipped_length"] = skipped
        summary["medqa"][f"{n_shot}-shot"] = result
        print(f"MEDQA {n_shot}-shot", json.dumps(result), flush=True)

    print("medqa cloze", flush=True)
    sums, means = cloze_scores(
        model, tokenizer, [(row["question"], [row["options"][letter] for letter in LETTERS]) for row in test],
    )
    summary["medqa"]["cloze"] = cloze_accuracy(sums, means, [LETTERS.index(row["answer_idx"]) for row in test])
    print("MEDQA cloze", json.dumps(summary["medqa"]["cloze"]), flush=True)

    print(f"medqa cloze {CLOZE_SHOTS}-shot", flush=True)
    prefix = shots_text([(q, options[LETTERS.index(a)]) for q, options, a in shots[:CLOZE_SHOTS]])
    sums, means = cloze_scores(
        model, tokenizer,
        [(row["question"], [row["options"][letter] for letter in LETTERS], prefix) for row in test],
    )
    summary["medqa"][f"cloze_{CLOZE_SHOTS}-shot"] = cloze_accuracy(
        sums, means, [LETTERS.index(row["answer_idx"]) for row in test],
    )
    print(f"MEDQA cloze {CLOZE_SHOTS}-shot", json.dumps(summary["medqa"][f"cloze_{CLOZE_SHOTS}-shot"]), flush=True)

    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)
    brief = {
        "mmlu_0": summary["mmlu"]["0-shot"]["overall"],
        "mmlu_5": summary["mmlu"]["5-shot"]["overall"],
        "medqa_0": summary["medqa"]["0-shot"],
        "medqa_5": summary["medqa"]["5-shot"],
        "mmlu_cloze": summary["mmlu"]["cloze"]["overall"],
        "medqa_cloze": summary["medqa"]["cloze"],
        f"mmlu_cloze_{CLOZE_SHOTS}": summary["mmlu"][f"cloze_{CLOZE_SHOTS}-shot"]["overall"],
        f"medqa_cloze_{CLOZE_SHOTS}": summary["medqa"][f"cloze_{CLOZE_SHOTS}-shot"],
    }
    print("DONE", json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
