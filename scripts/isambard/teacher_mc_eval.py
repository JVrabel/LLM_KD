#!/usr/bin/env python3
"""Multiple-choice scores for the frozen Phase 1 teacher.

Same layout the old medical MMLU run used: a subject line, five answered
examples, then the question with options A–D and the word Answer. The score is
which of the four letter tokens is most probable there. MedQA uses that same
layout, with five examples taken from its training file.
"""

import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/projects/u6wc/savants/models/Qwen3.5-35B-A3B-Base"
MMLU_DIR = "/projects/u6wc/savants/data/raw/eval/mmlu_medical"
MEDQA_TEST = "/projects/u6wc/savants/data/raw/eval/GBaker__MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl"
MEDQA_TRAIN = "/projects/u6wc/savants/data/raw/eval/GBaker__MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl"
OUT = "/projects/u6wc/savants/logs/teacher_mc_results.json"

SUBJECTS = {
    "anatomy": "anatomy",
    "clinical_knowledge": "clinical knowledge",
    "college_medicine": "college medicine",
    "medical_genetics": "medical genetics",
    "professional_medicine": "professional medicine",
}
LETTERS = "ABCD"
# Single tokens for " A", " B", " C", " D" in this tokenizer, and they stay
# single tokens when appended to a prompt that ends in "Answer:".
LETTER_IDS = {"A": 357, "B": 417, "C": 351, "D": 414}
MAX_LEN = 2048


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


def letter_logprobs(llm, tokenizer, prompts):
    """Log-probability of each answer letter as the token after ``Answer:``."""
    encoded = []
    kept = []
    for index, text in enumerate(prompts):
        prefix = tokenizer.encode(text, add_special_tokens=False)
        if len(prefix) + 1 > MAX_LEN:
            continue
        for letter in LETTERS:
            encoded.append({"prompt_token_ids": prefix + [LETTER_IDS[letter]]})
        kept.append(index)
    scoring = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    values = {}
    batch = 16
    for start in range(0, len(encoded), batch):
        results = llm.generate(encoded[start:start + batch], scoring)
        for offset, result in enumerate(results):
            entry = result.prompt_logprobs[-1] or {}
            token_id = result.prompt_token_ids[-1]
            chosen = entry.get(token_id)
            owner = kept[(start + offset) // 4]
            values.setdefault(owner, []).append(None if chosen is None else chosen.logprob)
        done = min(start + batch, len(encoded))
        print(f"scored {done}/{len(encoded)}", flush=True)
    return values, len(prompts) - len(kept)


def accuracy(rows, values, gold):
    correct = answered = 0
    for index, row in enumerate(rows):
        scores = values.get(index)
        if not scores or len(scores) != 4 or any(score is None for score in scores):
            continue
        answered += 1
        pick = LETTERS[max(range(4), key=lambda item: scores[item])]
        correct += int(pick == gold(row))
    return {
        "n": len(rows),
        "answered": answered,
        "correct": correct,
        "accuracy": round(correct / answered, 4) if answered else 0.0,
    }


def mmlu_rows():
    grouped = {}
    for key, label in SUBJECTS.items():
        dev = load_jsonl(os.path.join(MMLU_DIR, f"{key}_dev.jsonl"))
        test = load_jsonl(os.path.join(MMLU_DIR, f"{key}_test.jsonl"))
        shots = [(row["question"], row["choices"], LETTERS[row["answer"]]) for row in dev]
        grouped[label] = (shots, test)
    return grouped


def prompts_for(subject, shots, rows, choices_of, n_shot):
    used = shots[:n_shot]
    return [exam_prompt(subject, used, row["question"] if "question" in row else row, choices_of(row)) for row in rows]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    boundary = tokenizer.encode("Answer:", add_special_tokens=False)
    for letter, token_id in LETTER_IDS.items():
        full = tokenizer.encode(f"Answer: {letter}", add_special_tokens=False)
        if full != boundary + [token_id]:
            raise SystemExit(f"letter {letter} is not a single token after Answer:")

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=MAX_LEN,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        language_model_only=True,
        trust_remote_code=True,
        enforce_eager=True,
    )

    summary = {"mmlu": {}, "medqa": {}}
    grouped = mmlu_rows()
    flat_rows = []
    flat_prompts = {0: [], 5: []}
    flat_subjects = []
    for label, (shots, rows) in grouped.items():
        for n_shot in (0, 5):
            flat_prompts[n_shot].extend(
                prompts_for(label, shots, rows, lambda row: row["choices"], n_shot)
            )
        flat_rows.extend(rows)
        flat_subjects.extend([label] * len(rows))

    for n_shot in (0, 5):
        print(f"mmlu {n_shot}-shot", flush=True)
        values, skipped = letter_logprobs(llm, tokenizer, flat_prompts[n_shot])
        overall = accuracy(flat_rows, values, lambda row: LETTERS[row["answer"]])
        overall["skipped_length"] = skipped
        by_subject = {}
        for label in grouped:
            indexes = [i for i, subject in enumerate(flat_subjects) if subject == label]
            subset = [flat_rows[i] for i in indexes]
            remapped = {new: values[old] for new, old in enumerate(indexes) if old in values}
            by_subject[label] = accuracy(subset, remapped, lambda row: LETTERS[row["answer"]])
        summary["mmlu"][f"{n_shot}-shot"] = {"overall": overall, "subjects": by_subject}
        print(f"MMLU {n_shot}-shot", json.dumps(summary["mmlu"][f"{n_shot}-shot"]), flush=True)

    test = load_jsonl(MEDQA_TEST)
    train = load_jsonl(MEDQA_TRAIN, limit=5)
    shots = [
        (row["question"], [row["options"][letter] for letter in LETTERS], row["answer_idx"])
        for row in train
    ]

    def medqa_choices(row):
        return [row["options"][letter] for letter in LETTERS]

    for n_shot in (0, 5):
        print(f"medqa {n_shot}-shot", flush=True)
        prompts = prompts_for("medicine", shots, test, medqa_choices, n_shot)
        values, skipped = letter_logprobs(llm, tokenizer, prompts)
        result = accuracy(test, values, lambda row: row["answer_idx"])
        result["skipped_length"] = skipped
        summary["medqa"][f"{n_shot}-shot"] = result
        print(f"MEDQA {n_shot}-shot", json.dumps(result), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
