"""Drop Apollo question-answer rows that repeat a medical exam test question.

A row is dropped when the first 80 characters of its normalised question stem
(lower case, letters, digits and single spaces only) equal those of a question
in the MedQA-USMLE test set or one of the five medical MMLU test sets. The
same rule found 21 of 1273 MedQA and 8 of 945 MMLU questions in the file on
29 Oct 2026. Rows are [prompt, answer] lists; the stem is the text between
"Question:" and "Options:" in the prompt. Writes the surviving rows unchanged
and a json report listing every exam question that was matched.
"""

import argparse
import glob
import json
import os
import re

PREFIX = 80
MIN_LEN = 40


def normalise(text):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def stem(text):
    match = re.search(r"Question:(.*?)(?:\nOptions:|$)", text, re.S)
    return normalise(match.group(1) if match else text)


def exam_questions(medqa_test, mmlu_dir):
    questions = {}
    with open(medqa_test) as handle:
        for index, line in enumerate(handle):
            if line.strip():
                questions.setdefault(normalise(json.loads(line)["question"])[:PREFIX], f"medqa:{index}")
    for path in sorted(glob.glob(os.path.join(mmlu_dir, "*_test.jsonl"))):
        subject = os.path.basename(path).replace("_test.jsonl", "")
        with open(path) as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    questions.setdefault(normalise(json.loads(line)["question"])[:PREFIX], f"mmlu/{subject}:{index}")
    return {key: name for key, name in questions.items() if len(key) >= MIN_LEN}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--medqa-test", required=True)
    parser.add_argument("--mmlu-dir", required=True)
    args = parser.parse_args()

    questions = exam_questions(args.medqa_test, args.mmlu_dir)
    print(f"{len(questions)} exam question prefixes", flush=True)
    kept = dropped = 0
    matched = {}
    with open(args.jsonl) as source, open(args.out, "w") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row[0] if isinstance(row, list) else row.get("text", "")
            key = stem(text)[:PREFIX]
            name = questions.get(key) if len(key) >= MIN_LEN else None
            if name is None:
                sink.write(line if line.endswith("\n") else line + "\n")
                kept += 1
            else:
                matched.setdefault(name, 0)
                matched[name] += 1
                dropped += 1
    report = {
        "source": args.jsonl,
        "out": args.out,
        "prefix_chars": PREFIX,
        "kept": kept,
        "dropped": dropped,
        "exam_questions_hit": len(matched),
        "medqa_hit": sum(1 for name in matched if name.startswith("medqa")),
        "mmlu_hit": sum(1 for name in matched if name.startswith("mmlu")),
        "rows_dropped_per_question": matched,
    }
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rows_dropped_per_question"}), flush=True)


if __name__ == "__main__":
    main()
