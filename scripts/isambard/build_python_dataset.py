#!/usr/bin/env python3
"""Build a clean Python training set from HuggingFaceTB/stack-edu.

Selection (in order):
  1. Permissively licensed files only (MIT, Apache, BSD, ...).
  2. Highest educational-score files first, until TOKEN_BUDGET tokens
     (counted with the Qwen3 tokenizer).
  3. Decontamination against HumanEval and MBPP: drop any file that shares an
     n-word span with a benchmark problem statement or reference solution.
  4. Near-duplicate removal by 64-bit SimHash (files within a small Hamming
     distance of an already kept file are dropped).

File contents are fetched from the public Software Heritage S3 bucket
(s3://softwareheritage/content/<blob_id>), as required by stack-edu.

Output (JSONL, one document per line: text, repo, path, score, license, tokens):
  <out>/python_clean.jsonl           the dataset
  <out>/python_clean.stats.json      counts per stage
  <out>/python_clean.removed.jsonl   files dropped by decontamination, with the
                                     matched benchmark span (for manual review)
"""

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import botocore
from botocore import UNSIGNED
from datasets import load_dataset
from transformers import AutoTokenizer

N_WORDS = 10
TOKEN_BUDGET = 2_000_000_000
SIMHASH_BITS = 64
SIMHASH_MAX_DISTANCE = 3
MIN_TOKENS = 32
MAX_TOKENS = 16384
BUCKET = "softwareheritage"

# Benchmark text that is too generic to be evidence of contamination.
HUMAN_EVAL_SOLUTIONS_OK = {"return x + y", "return len(string)", "return n**2", "return ''.join(strings)"}

WORD_RE = re.compile(r"\w+")


def word_spans(text, n):
    words = WORD_RE.findall(text.lower())
    # Require two words of 5+ characters. Spans of numbers and one-letter
    # variable names ("while n 1 if n 2 0 n n 2") collide between unrelated
    # solutions of the same small problem; copied code shares real identifiers.
    return {
        " ".join(words[i : i + n])
        for i in range(len(words) - n + 1)
        if sum(len(word) >= 5 for word in words[i : i + n]) >= 2
    }


def simhash(text):
    weights = [0] * SIMHASH_BITS
    for word in WORD_RE.findall(text.lower()):
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        for bit in range(SIMHASH_BITS):
            weights[bit] += 1 if value & (1 << bit) else -1
    out = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            out |= 1 << bit
    return out


class SimHashIndex:
    """Buckets by each 16-bit slice; a near-duplicate (distance <= 3) collides in at least one slice."""

    def __init__(self):
        self.buckets = [collections.defaultdict(list) for _ in range(4)]
        self.count = 0

    def too_close(self, value):
        for part in range(4):
            for other in self.buckets[part][(value >> (16 * part)) & 0xFFFF]:
                if (value ^ other).bit_count() <= SIMHASH_MAX_DISTANCE:
                    return True
        return False

    def add(self, value):
        for part in range(4):
            self.buckets[part][(value >> (16 * part)) & 0xFFFF].append(value)
        self.count += 1


def load_benchmarks():
    """Only problem statements, not reference solutions. Solutions to small
    algorithmic exercises (primes, Fibonacci, min/max) are independently
    rewritten by thousands of people, so matching them deletes original work.
    A copied benchmark problem is caught by its statement, which is distinctive."""
    spans = {}
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    for row in humaneval:
        prompt = row["prompt"]
        statement = prompt.split('"""')[1] if prompt.count('"""') >= 2 else prompt
        spans.setdefault("humaneval_statement", set()).update(word_spans(statement, N_WORDS))

    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for row in mbpp:
        spans.setdefault("mbpp_statement", set()).update(word_spans(row["prompt"], N_WORDS))

    banned = set().union(*spans.values())
    print(f"benchmark spans: {len(banned)} unique ({', '.join(f'{k}={len(v)}' for k, v in spans.items())})", flush=True)
    return banned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--max-files", type=int, default=0, help="stop after scanning this many files (0 = no limit)")
    parser.add_argument("--min-score", type=float, default=3.0, help="keep files at or above this educational score")
    parser.add_argument("--workers", type=int, default=32, help="parallel S3 downloads")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading stack-edu Python metadata", flush=True)
    meta = load_dataset("HuggingFaceTB/stack-edu", "Python", split="train")
    rows = meta.filter(lambda r: r["license_type"] == "permissive" and r["score"] >= args.min_score)
    cutoff = args.min_score
    rows = rows.sort(["score", "length_bytes"], reverse=[True, True])
    print(f"permissive files: {len(rows)} of {len(meta)}, score cutoff {cutoff:.3f}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    banned = load_benchmarks()

    s3 = boto3.client("s3", config=botocore.config.Config(signature_version=UNSIGNED, max_pool_connections=args.workers))

    def fetch(blob_id):
        obj = s3.get_object(Bucket=BUCKET, Key=f"content/{blob_id}")
        return gzip.decompress(obj["Body"].read()).decode("utf-8", errors="ignore")

    stats = collections.Counter()
    kept_tokens = 0
    index = SimHashIndex()
    done_blobs = set()
    started = time.time()

    out_path = os.path.join(args.out, "python_clean.jsonl")
    removed_path = os.path.join(args.out, "python_clean.removed.jsonl")
    seen_path = os.path.join(args.out, "python_clean.seen.txt")
    if os.path.exists(seen_path):
        with open(seen_path) as f:
            done_blobs = {line.strip() for line in f if line.strip()}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                record = json.loads(line)
                kept_tokens += record["tokens"]
                index.add(simhash(record["text"]))
                stats["kept"] += 1
    print(f"resume: {len(done_blobs)} files already scanned, {stats['kept']} kept, "
          f"{kept_tokens/1e9:.3f}B tokens", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)

    def handle_batch(batch, out, removed, seen):
        """Download a batch concurrently, then judge files in score order."""
        nonlocal kept_tokens
        fetched = {}

        def fetch_one(row):
            try:
                return row["blob_id"], fetch(row["blob_id"]), None
            except Exception as exc:
                return row["blob_id"], None, exc

        for blob_id, text, exc in pool.map(fetch_one, batch):
            fetched[blob_id] = (text, exc)
        for row in batch:
            text, exc = fetched[row["blob_id"]]
            seen.write(row["blob_id"] + "\n")
            if exc is not None:
                stats["download_failed"] += 1
                if stats["download_failed"] <= 5:
                    print(f"download failed: {row['blob_id']} ({exc})", flush=True)
                continue
            judge(row, text, out, removed)

    def judge(row, text, out, removed):
        nonlocal kept_tokens
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if tokens < MIN_TOKENS or tokens > MAX_TOKENS:
            stats["bad_length"] += 1
            return

        overlap = word_spans(text, N_WORDS) & banned
        if overlap:
            stats["contaminated"] += 1
            removed.write(json.dumps({
                "blob_id": row["blob_id"], "repo": row["repo_name"], "path": row["path"],
                "score": row["score"], "matched_span": sorted(overlap)[0],
            }) + "\n")
            return

        fingerprint = simhash(text)
        if index.too_close(fingerprint):
            stats["near_duplicate"] += 1
            return
        index.add(fingerprint)

        kept_tokens += tokens
        stats["kept"] += 1
        out.write(json.dumps({
            "blob_id": row["blob_id"], "text": text, "repo": row["repo_name"],
            "path": row["path"], "score": row["score"],
            "license": row["detected_licenses"], "tokens": tokens,
        }) + "\n")

        if stats["kept"] % 5000 == 0:
            rate = stats["seen"] / (time.time() - started)
            print(f"kept {stats['kept']} files, {kept_tokens/1e9:.3f}B tokens, "
                  f"seen {stats['seen']} ({rate:.0f} files/s)", flush=True)

    with open(out_path, "a") as out, open(removed_path, "a") as removed, open(seen_path, "a") as seen:
        batch = []
        for row in rows:
            if kept_tokens >= args.token_budget or (args.max_files and stats["seen"] >= args.max_files):
                break
            stats["seen"] += 1
            if row["blob_id"] in done_blobs:
                stats["skipped_done"] += 1
                continue
            batch.append(row)
            if len(batch) >= args.workers * 8:
                handle_batch(batch, out, removed, seen)
                batch = []
                if kept_tokens >= args.token_budget:
                    break
        if batch and kept_tokens < args.token_budget:
            handle_batch(batch, out, removed, seen)
    pool.shutdown()

    stats["tokens"] = kept_tokens
    with open(os.path.join(args.out, "python_clean.stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("DONE", dict(stats), flush=True)


if __name__ == "__main__":
    sys.exit(main())
