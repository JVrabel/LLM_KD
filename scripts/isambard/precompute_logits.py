#!/usr/bin/env python3
"""Precompute the Phase 1 teacher's top-64 next-token log-probabilities.

One ``.ids.npy`` and ``.lp.npy`` pair is written per packed shard, position-aligned
with the shard: index t is the distribution of token t. Run with the vLLM 0.17
environment on two GPUs, which is the configuration that fits this model.
"""

import argparse
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
            # vLLM can include the observed prompt token in addition to the
            # requested top-k, so explicitly retain the highest 64 entries.
            ranked = sorted(entry.items(), key=lambda item: -item[1].logprob)[:TOPK]
            size = len(ranked)
            out_ids[row, position, :size] = np.fromiter(
                (token_id for token_id, _ in ranked), dtype=np.uint32, count=size,
            )
            out_lp[row, position, :size] = np.fromiter(
                (value.logprob for _, value in ranked), dtype=np.float16, count=size,
            )
    return out_ids, out_lp


def resume_index(lp_path, count, seq_len):
    """First unscored row, ``count`` when every row is scored, or None if the file cannot be reused.

    Position 0 has no next-token distribution. A row is scored when position 1
    is no longer the sentinel written at creation.
    """
    try:
        values = np.load(lp_path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if values.shape != (count, seq_len, TOPK) or values.dtype != np.float16 or seq_len < 2:
        return None
    column = np.asarray(values[:, 1, 0])
    pending = np.flatnonzero(column == SENTINEL)
    if pending.size == 0:
        return count
    start = int(pending[0])
    if np.any(column[start:] != SENTINEL):
        return None
    return start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means every shard")
    parser.add_argument("--batch-sequences", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.shards, "meta.json")) as handle:
        meta = json.load(handle)
    seq_len = meta["seq_len"]
    n_shards = len(meta["shards"])
    if args.max_shards:
        n_shards = min(n_shards, args.max_shards)

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

    for number in range(n_shards):
        ids_path = os.path.join(args.out, f"shard-{number:05d}.ids.npy")
        lp_path = os.path.join(args.out, f"shard-{number:05d}.lp.npy")
        done_path = os.path.join(args.out, f"shard-{number:05d}.done")
        if os.path.exists(done_path):
            print(f"shard {number} exists", flush=True)
            continue
        count = meta["shards"][number]
        shape = (count, seq_len, TOPK)
        start_row = resume_index(lp_path, count, seq_len) if os.path.exists(lp_path) else None
        ids_ok = False
        if start_row is not None and os.path.exists(ids_path):
            try:
                ids = np.load(ids_path, mmap_mode="r")
                ids_ok = ids.shape == shape and ids.dtype == np.uint32
                del ids
            except (OSError, ValueError):
                ids_ok = False
        if start_row is None or not ids_ok:
            for path in (ids_path, lp_path):
                if os.path.exists(path):
                    os.remove(path)
            start_row = 0
            out_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.uint32, shape=shape)
            out_lp = np.lib.format.open_memmap(lp_path, mode="w+", dtype=np.float16, shape=shape)
            out_lp[:] = SENTINEL
        else:
            out_ids = np.lib.format.open_memmap(ids_path, mode="r+", dtype=np.uint32, shape=shape)
            out_lp = np.lib.format.open_memmap(lp_path, mode="r+", dtype=np.float16, shape=shape)
        if start_row >= count:
            open(done_path, "w").close()
            print(f"shard {number} already scored", flush=True)
            continue
        if start_row:
            print(f"shard {number} resume at {start_row}/{count}", flush=True)
        tokens = np.memmap(
            os.path.join(args.shards, f"shard-{number:05d}.bin"),
            dtype=np.uint32, mode="r", shape=(count, seq_len),
        )
        started = time.perf_counter()
        for start in range(start_row, count, args.batch_sequences):
            stop = min(start + args.batch_sequences, count)
            prompts = [{"prompt_token_ids": tokens[row].tolist()} for row in range(start, stop)]
            results = llm.generate(prompts, scoring)
            batch_ids, batch_lp = pack_results(results, seq_len)
            out_ids[start:stop] = batch_ids
            out_lp[start:stop] = batch_lp
            elapsed = time.perf_counter() - started
            rate = (stop - start_row) * seq_len / elapsed
            print(f"shard {number} sequences {stop}/{count} rate {rate:.0f} tok/s", flush=True)
        out_ids.flush()
        out_lp.flush()
        open(done_path, "w").close()
    print("DONE", json.dumps({"shards": n_shards, "topk": TOPK}), flush=True)


if __name__ == "__main__":
    main()
