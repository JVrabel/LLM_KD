#!/usr/bin/env python3
"""Precompute the Phase 1 teacher's top-64 next-token log-probabilities.

One ``.ids.npy`` and ``.lp.npy`` pair is written per packed shard, position-aligned
with the shard: index t is the distribution of token t. Run with the vLLM 0.17
environment on two GPUs, which is the configuration that fits this model.
"""

import argparse
import json
import os

import numpy as np
from vllm import LLM, SamplingParams

TOPK = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means every shard")
    parser.add_argument("--batch-sequences", type=int, default=4)
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
        max_model_len=seq_len,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        language_model_only=True,
        trust_remote_code=True,
        enforce_eager=True,
    )
    scoring = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=TOPK)

    for number in range(n_shards):
        ids_path = os.path.join(args.out, f"shard-{number:05d}.ids.npy")
        lp_path = os.path.join(args.out, f"shard-{number:05d}.lp.npy")
        if os.path.exists(ids_path) and os.path.exists(lp_path):
            print(f"shard {number} exists", flush=True)
            continue
        count = meta["shards"][number]
        tokens = np.memmap(
            os.path.join(args.shards, f"shard-{number:05d}.bin"),
            dtype=np.uint32, mode="r", shape=(count, seq_len),
        )
        out_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.uint32, shape=(count, seq_len, TOPK))
        out_lp = np.lib.format.open_memmap(lp_path, mode="w+", dtype=np.float16, shape=(count, seq_len, TOPK))
        out_lp[:] = np.float16(-1e4)
        for start in range(0, count, args.batch_sequences):
            stop = min(start + args.batch_sequences, count)
            prompts = [{"prompt_token_ids": tokens[row].tolist()} for row in range(start, stop)]
            results = llm.generate(prompts, scoring)
            for row, result in enumerate(results):
                for position, entry in enumerate(result.prompt_logprobs):
                    if not entry:
                        continue
                    ranked = sorted(entry.items(), key=lambda item: -item[1].logprob)[:TOPK]
                    for slot, (token_id, logprob) in enumerate(ranked):
                        out_ids[start + row, position, slot] = token_id
                        out_lp[start + row, position, slot] = logprob.logprob
            print(f"shard {number} sequences {stop}/{count}", flush=True)
        out_ids.flush()
        out_lp.flush()
    print("DONE", json.dumps({"shards": n_shards, "topk": TOPK}), flush=True)


if __name__ == "__main__":
    main()
