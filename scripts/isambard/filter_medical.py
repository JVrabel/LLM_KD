#!/usr/bin/env python3
"""Build the medical training text from PubMed abstracts and Apollo.

Drops a PubMed abstract when any of these hold:

- its normalized text is an exact duplicate of another kept document
- two or more of its 12-word windows occur in Apollo (the abstract is contained
  in a document we already have)
- it shares a 10-word span with a medical eval question, answer, or passage

Apollo is kept except for exact duplicates and eval contamination. Output is
JSONL with a ``text`` field, ready for ``pack_shards.py``.
"""

import argparse
import glob
import hashlib
import json
import os
import re

from datasets import load_dataset

WORD_RE = re.compile(r"\w+")
WINDOW = 12
STRIDE = 40
NGRAM = 10
MIN_CHARS = 200
MEDICAL_MMLU = (
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def words_of(text):
    return WORD_RE.findall(text.lower())


def digest(text):
    return hashlib.blake2b(text.encode(), digest_size=8).digest()


def window_hashes(text):
    words = words_of(text)
    if len(words) < WINDOW:
        return
    for start in range(0, len(words) - WINDOW + 1, STRIDE):
        yield digest(" ".join(words[start : start + WINDOW]))


def ngram_spans(text):
    words = words_of(text)
    spans = set()
    for start in range(len(words) - NGRAM + 1):
        span = words[start : start + NGRAM]
        if sum(len(word) >= 5 for word in span) >= 2:
            spans.add(" ".join(span))
    return spans


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings_in(item)


def load_rows(path, config=None):
    try:
        if config:
            return load_dataset(path, config, split="train")
        return load_dataset(path, split="train")
    except Exception:
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if not files:
            raise
        return load_dataset("parquet", data_files=files, split="train")


def eval_spans(eval_dir):
    banned = set()
    sources = [
        ("GBaker__MedQA-USMLE-4-options", None, ("question", "answer", "options")),
        ("openlifescienceai__medmcqa", None, ("question", "opa", "opb", "opc", "opd", "exp")),
        ("qiaojin__PubMedQA", "pqa_labeled", ("question", "long_answer", "context")),
    ]
    for name, config, fields in sources:
        path = os.path.join(eval_dir, name)
        rows = load_rows(path, config)
        for row in rows:
            for field in fields:
                if field not in row:
                    continue
                for text in strings_in(row[field]):
                    banned |= ngram_spans(text)
        print(f"{name}: banned spans now {len(banned)}", flush=True)

    mmlu = os.path.join(eval_dir, "cais__mmlu")
    for subject in MEDICAL_MMLU:
        try:
            rows = load_rows(mmlu, subject)
        except Exception as exc:
            print(f"mmlu/{subject} skipped ({exc})", flush=True)
            continue
        for row in rows:
            for text in strings_in(row.get("question")):
                banned |= ngram_spans(text)
            for text in strings_in(row.get("choices")):
                banned |= ngram_spans(text)
        print(f"mmlu/{subject}: banned spans now {len(banned)}", flush=True)
    return banned


def contaminated(text, banned):
    words = words_of(text)
    for start in range(len(words) - NGRAM + 1):
        span = words[start : start + NGRAM]
        if sum(len(word) >= 5 for word in span) < 2:
            continue
        if " ".join(span) in banned:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubmed-dir", required=True)
    parser.add_argument("--apollo", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        print(f"already filtered: {args.out}", flush=True)
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("loading eval spans", flush=True)
    banned = eval_spans(args.eval_dir)

    print("indexing Apollo", flush=True)
    apollo_exact = set()
    apollo_windows = set()
    stats = {"apollo_seen": 0, "apollo_dup": 0, "apollo_contaminated": 0, "apollo_kept": 0}
    tmp = args.out + ".partial"
    out = open(tmp, "w")
    with open(args.apollo) as handle:
        for line in handle:
            text = json.loads(line)["text"].strip()
            stats["apollo_seen"] += 1
            key = digest(normalize(text))
            if key in apollo_exact:
                stats["apollo_dup"] += 1
                continue
            if contaminated(text, banned):
                stats["apollo_contaminated"] += 1
                continue
            apollo_exact.add(key)
            apollo_windows.update(window_hashes(text))
            stats["apollo_kept"] += 1
            out.write(json.dumps({"text": text, "source": "apollo"}) + "\n")
            if stats["apollo_seen"] % 50000 == 0:
                print(f"apollo {stats['apollo_seen']} windows {len(apollo_windows)}", flush=True)
    print(f"apollo index {len(apollo_windows)} windows", flush=True)

    stats.update({"pubmed_seen": 0, "pubmed_short": 0, "pubmed_dup": 0,
                  "pubmed_in_apollo": 0, "pubmed_contaminated": 0, "pubmed_kept": 0})
    seen_exact = set(apollo_exact)
    for path in sorted(glob.glob(os.path.join(args.pubmed_dir, "*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                text = json.loads(line)["text"].strip()
                stats["pubmed_seen"] += 1
                if len(text) < MIN_CHARS:
                    stats["pubmed_short"] += 1
                    continue
                key = digest(normalize(text))
                if key in seen_exact:
                    stats["pubmed_dup"] += 1
                    continue
                hits = sum(1 for item in window_hashes(text) if item in apollo_windows)
                if hits >= 2:
                    stats["pubmed_in_apollo"] += 1
                    continue
                if contaminated(text, banned):
                    stats["pubmed_contaminated"] += 1
                    continue
                seen_exact.add(key)
                stats["pubmed_kept"] += 1
                out.write(json.dumps({"text": text, "source": "pubmed"}) + "\n")
        print(f"{os.path.basename(path)} kept {stats['pubmed_kept']}", flush=True)
    out.close()
    os.rename(tmp, args.out)
    stats_path = args.out + ".stats.json"
    with open(stats_path, "w") as handle:
        json.dump(stats, handle, indent=2)
    print("DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
