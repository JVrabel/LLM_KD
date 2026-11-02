#!/usr/bin/env python3
"""Benchmark vLLM teacher scoring batches and Python result packing.

This reads a few packed Python sequences and does not write logit shards. It
uses the same model and prompt-logprob settings as precompute_logits.py.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
from vllm import LLM, SamplingParams

TOPK = 64
SENTINEL = np.float16(-1e4)


def pack_results(results, seq_len):
    """Convert vLLM's per-token dictionaries to dense arrays in bulk."""
    out_ids = np.zeros((len(results), seq_len, TOPK), dtype=np.uint32)
    out_lp = np.full((len(results), seq_len, TOPK), SENTINEL, dtype=np.float16)
    for row, result in enumerate(results):
        for position, entry in enumerate(result.prompt_logprobs):
            if not entry:
                continue
            ranked = sorted(entry.items(), key=lambda item: -item[1].logprob)[:TOPK]
            size = len(ranked)
            out_ids[row, position, :size] = np.fromiter(
                (token_id for token_id, _ in ranked), dtype=np.uint32, count=size,
            )
            out_lp[row, position, :size] = np.fromiter(
                (value.logprob for _, value in ranked), dtype=np.float16, count=size,
            )
    return out_ids, out_lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--sequences", type=int, default=16)
    args = parser.parse_args()

    with open(os.path.join(args.shards, "meta.json")) as handle:
        meta = json.load(handle)
    seq_len = meta["seq_len"]
    count = meta["shards"][0]
    tokens = np.memmap(
        os.path.join(args.shards, "shard-00000.bin"),
        dtype=np.uint32,
        mode="r",
        shape=(count, seq_len),
    )
    prompts = [
        {"prompt_token_ids": tokens[row].tolist()}
        for row in range(min(args.sequences, count))
    ]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=seq_len + 1,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.75,
        language_model_only=True,
        trust_remote_code=True,
        enforce_eager=True,
        max_logprobs=TOPK,
    )
    scoring = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=TOPK)

    summary = []
    for batch_size in args.batch_sizes:
        inference_seconds = 0.0
        packing_seconds = 0.0
        scored = 0
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            before = time.perf_counter()
            results = llm.generate(chunk, scoring, use_tqdm=False)
            inference_seconds += time.perf_counter() - before
            before = time.perf_counter()
            ids, logprobs = pack_results(results, seq_len)
            packing_seconds += time.perf_counter() - before
            scored += len(chunk)
            # Force materialization before releasing the arrays and Python objects.
            assert logprobs[:, 1, 0].max() > SENTINEL
            assert ids.shape == (len(chunk), seq_len, TOPK)
            del ids, logprobs, results
            gc.collect()
        token_count = scored * seq_len
        result = {
            "batch_sequences": batch_size,
            "sequences": scored,
            "tokens": token_count,
            "inference_seconds": round(inference_seconds, 3),
            "packing_seconds": round(packing_seconds, 3),
            "tokens_per_second": round(token_count / (inference_seconds + packing_seconds), 1),
        }
        summary.append(result)
        print("BENCH", json.dumps(result), flush=True)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
