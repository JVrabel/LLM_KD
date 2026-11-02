#!/usr/bin/env python3
"""HumanEval pass@1 for a from-scratch Qwen3 student checkpoint.

Same protocol as the teacher exam: greedy completion, at most 256 new tokens,
stop at a top-level ``class`` / ``def`` / comment / ``if`` / ``print``, then run
the official tests in a short-lived subprocess. The checkpoint is the state
dict written by ``src/train_kd.py``.
"""

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys

import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

CHECKPOINT = "/projects/u6wc/savants/runs/pilot_0.6b_ntp/step_0030517.pt"
TOKENIZER = "/projects/u6wc/savants/models/Qwen3.5-35B-A3B-Base"
HUMANEVAL = "/projects/u6wc/savants/data/raw/eval/humaneval/HumanEval.jsonl"
EOS = 248044
MAX_NEW_TOKENS = 256
BATCH = 8
STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


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


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trim(text):
    cut = len(text)
    for stop in STOPS:
        found = text.find(stop)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]


class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, prompt_width):
        self.tokenizer = tokenizer
        self.prompt_width = prompt_width

    def __call__(self, input_ids, scores, **kwargs):
        flags = []
        for row in input_ids:
            text = self.tokenizer.decode(row[self.prompt_width:], skip_special_tokens=True)
            flags.append(any(stop in text for stop in STOPS))
        return torch.tensor(flags, device=input_ids.device)


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


def complete(model, tokenizer, prompts):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    width = encoded["input_ids"].shape[1]
    criteria = StoppingCriteriaList([StopOnStrings(tokenizer, width)])
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=EOS,
            pad_token_id=EOS,
            stopping_criteria=criteria,
        )
    texts = []
    for row in output:
        texts.append(trim(tokenizer.decode(row[width:], skip_special_tokens=True)))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--humaneval", default=HUMANEVAL)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    records = load_jsonl(args.humaneval)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token_id = EOS
    tokenizer.padding_side = "left"
    model = load_model(args.checkpoint)
    print(f"humaneval problems {len(records)} checkpoint {args.checkpoint}", flush=True)

    passed = 0
    details = []
    for start in range(0, len(records), BATCH):
        rows = records[start:start + BATCH]
        completions = complete(model, tokenizer, [row["prompt"] for row in rows])
        for row, completion in zip(rows, completions):
            program = row["prompt"] + completion + "\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
            ok = passes(program)
            passed += int(ok)
            details.append({"task_id": row["task_id"], "passed": ok, "completion": completion})
        done = min(start + BATCH, len(records))
        print(f"humaneval {done}/{len(records)} passed {passed}", flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "n": len(records),
        "passed": passed,
        "pass_at_1": round(passed / len(records), 4) if records else 0.0,
        "details": details,
    }
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("HUMANEVAL", json.dumps({key: summary[key] for key in ("n", "passed", "pass_at_1")}), flush=True)


if __name__ == "__main__":
    main()
