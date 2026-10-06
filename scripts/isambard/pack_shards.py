#!/usr/bin/env python3
"""Pack a JSONL corpus into fixed-length token shards.

Documents are tokenized with the student tokenizer, separated by the end-of-sequence
token, and concatenated into sequences of ``seq_len`` tokens. The remainder of the
last sequence is dropped. Output is little-endian uint32, one file per shard:

    shard-00000.bin    shape (n_seq, seq_len)
    meta.json          seq_len, eos id, per-shard sequence counts
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer


def documents(paths):
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)["text"].strip()
                if text:
                    yield text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--shard-seqs", type=int, default=8192)
    args = parser.parse_args()

    meta_path = os.path.join(args.out, "meta.json")
    if os.path.exists(meta_path):
        print(f"already packed: {meta_path}", flush=True)
        return

    os.makedirs(args.out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id")

    buffer = []
    shard_counts = []
    sequences = 0
    docs = 0
    block = np.empty((args.shard_seqs, args.seq_len), dtype=np.uint32)
    filled = 0

    def flush():
        nonlocal filled
        if filled == 0:
            return
        name = f"shard-{len(shard_counts):05d}.bin"
        block[:filled].tofile(os.path.join(args.out, name))
        shard_counts.append(filled)
        filled = 0

    for text in documents(args.jsonl):
        docs += 1
        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids)
        buffer.append(eos)
        while len(buffer) >= args.seq_len:
            block[filled] = buffer[: args.seq_len]
            del buffer[: args.seq_len]
            filled += 1
            sequences += 1
            if filled == args.shard_seqs:
                flush()
        if docs % 20000 == 0:
            print(f"docs {docs}  sequences {sequences}", flush=True)

    flush()
    meta = {
        "seq_len": args.seq_len,
        "eos_token_id": eos,
        "dtype": "uint32",
        "documents": docs,
        "sequences": sequences,
        "dropped_tokens": len(buffer),
        "shards": shard_counts,
    }
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
