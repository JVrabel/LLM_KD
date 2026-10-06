#!/usr/bin/env python3
"""Task scores for the frozen Phase 1 teacher.

MedQA-USMLE test: each option is scored by its mean next-token log-probability
given the question, and the highest-scoring option wins. HumanEval: greedy
completion, pass@1, tests run in a short-lived subprocess.
"""

import argparse
import json
import os
import subprocess
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/projects/u6wc/savants/models/Qwen3.5-35B-A3B-Base"
MEDQA = "/projects/u6wc/savants/data/raw/eval/GBaker__MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl"
HUMANEVAL = "/projects/u6wc/savants/data/raw/eval/humaneval/HumanEval.jsonl"
MAX_LEN = 2048
LETTERS = ("A", "B", "C", "D")


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_logprob(result, option_start):
    total = 0.0
    count = 0
    for index, entry in enumerate(result.prompt_logprobs):
        if index < option_start or not entry:
            continue
        token_id = result.prompt_token_ids[index]
        chosen = entry.get(token_id)
        if chosen is None:
            return None
        total += chosen.logprob
        count += 1
    if count == 0:
        return None
    return total / count


def score_medqa(llm, tokenizer, records):
    prompts = []
    owners = []
    skipped = 0
    for index, row in enumerate(records):
        question = tokenizer.encode(row["question"].strip() + "\n", add_special_tokens=False)
        pieces = []
        fits = True
        for letter in LETTERS:
            option = tokenizer.encode(row["options"][letter].strip(), add_special_tokens=False)
            if not option or len(question) + len(option) > MAX_LEN:
                fits = False
                break
            pieces.append(question + option)
        if not fits or row.get("answer_idx") not in LETTERS:
            skipped += 1
            continue
        for piece in pieces:
            prompts.append({"prompt_token_ids": piece})
            owners.append((index, len(question)))
    scoring = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    scores = {}
    batch = 32
    done = 0
    for start in range(0, len(prompts), batch):
        results = llm.generate(prompts[start:start + batch], scoring)
        for result, (owner, option_start) in zip(results, owners[start:start + batch]):
            scores.setdefault(owner, []).append(mean_logprob(result, option_start))
        done += len(results)
        print(f"medqa scored {done}/{len(prompts)}", flush=True)

    correct = 0
    answered = 0
    for index, row in enumerate(records):
        values = scores.get(index)
        if not values or any(value is None for value in values) or len(values) != 4:
            continue
        answered += 1
        pick = LETTERS[max(range(4), key=lambda item: values[item])]
        correct += int(pick == row["answer_idx"])
    accuracy = correct / answered if answered else 0.0
    summary = {"answered": answered, "correct": correct, "accuracy": round(accuracy, 4), "skipped": skipped}
    print("MEDQA", json.dumps(summary), flush=True)
    return summary


def passes(program):
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            timeout=10,
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def score_humaneval(llm, tokenizer, records):
    prompts = [
        {"prompt_token_ids": tokenizer.encode(row["prompt"], add_special_tokens=False)}
        for row in records
    ]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
    )
    results = llm.generate(prompts, sampling)
    passed = 0
    for index, (row, result) in enumerate(zip(records, results), start=1):
        program = row["prompt"] + result.outputs[0].text + "\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
        ok = passes(program)
        passed += int(ok)
        if index % 20 == 0 or index == len(records):
            print(f"humaneval {index}/{len(records)} passed {passed}", flush=True)
    summary = {"n": len(records), "passed": passed, "pass_at_1": round(passed / len(records), 4) if records else 0.0}
    print("HUMANEVAL", json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
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
    summary = {
        "medqa": score_medqa(llm, tokenizer, load_jsonl(MEDQA)),
        "humaneval": score_humaneval(llm, tokenizer, load_jsonl(HUMANEVAL)),
    }
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
