#!/usr/bin/env python3
"""Pack a JSONL corpus into fixed-length token shards.

Documents are tokenized with the student tokenizer, separated by the end-of-sequence
token, and concatenated into sequences of ``seq_len`` tokens. The remainder of the
last sequence is dropped. Output is little-endian uint32, one file per shard:

    shard-00000.bin    shape (n_seq, seq_len)
    meta.json          seq_len, eos id, per-shard sequence counts

A row is ``{"text": ...}`` or a JSON list of strings (the Apollo question-answer
file), which is joined with newlines into one document.

``--shuffle-seed`` writes the documents in a seeded random order instead of file
order. The raw corpora are grouped by source and sorted by score, so without it
each shard is a different slice of the corpus and a run sees them one after
another; the October 2026 pilots measurably forgot early shards. Those packs
were written in file order and their teacher logits are aligned to them, so
they are not repacked: the trainer's ``training.shuffle`` draws their sequences
in random order instead. New packs should use this flag.
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer


def row_text(line):
    row = json.loads(line)
    if isinstance(row, list):
        return "\n".join(str(part).strip() for part in row).strip()
    return row["text"].strip()


def line_offsets(paths):
    """Byte offset of every non-blank line, as (file index, offset) arrays."""
    files, offsets = [], []
    for file_index, path in enumerate(paths):
        offset = 0
        with open(path, "rb") as handle:
            for line in handle:
                if line.strip():
                    files.append(file_index)
                    offsets.append(offset)
                offset += len(line)
        print(f"indexed {path}: {len(offsets)} lines so far", flush=True)
    return np.asarray(files, dtype=np.int32), np.asarray(offsets, dtype=np.int64)


def documents(paths, shuffle_seed):
    if shuffle_seed is None:
        for path in paths:
            with open(path) as handle:
                for line in handle:
                    if line.strip():
                        text = row_text(line)
                        if text:
                            yield text
        return
    files, offsets = line_offsets(paths)
    order = np.random.default_rng(shuffle_seed).permutation(len(offsets))
    handles = [open(path, "rb") for path in paths]
    try:
        for k in order:
            handle = handles[files[k]]
            handle.seek(int(offsets[k]))
            text = row_text(handle.readline())
            if text:
                yield text
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--shard-seqs", type=int, default=8192)
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="write documents in a seeded random order; default is file order")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="stop after this many full shards (a random subset with --shuffle-seed)")
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

    stopped = False
    for text in documents(args.jsonl, args.shuffle_seed):
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
                if args.max_shards and len(shard_counts) >= args.max_shards:
                    stopped = True
                    break
        if stopped:
            print(f"stopped after {len(shard_counts)} shards", flush=True)
            break
        if docs % 20000 == 0:
            print(f"docs {docs}  sequences {sequences}", flush=True)

    if not stopped:
        flush()
    meta = {
        "seq_len": args.seq_len,
        "eos_token_id": eos,
        "dtype": "uint32",
        "documents": docs,
        "sequences": sequences,
        "dropped_tokens": len(buffer),
        "shards": shard_counts,
        "sources": args.jsonl,
        "shuffle_seed": args.shuffle_seed,
    }
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
