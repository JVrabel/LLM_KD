#!/usr/bin/env python3
"""Loss of successive checkpoints on fixed rows from several shards.

The packed shards were written in corpus order, so each shard is a different
slice of the corpus. If the final checkpoint scores early shards worse than an
intermediate checkpoint did, the run forgot that part of the data.
"""

import argparse
import gc
import importlib.util
import json
import math
import os

import numpy as np
import torch

EOS = 248044


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


def rows_from(packed, shard, row0, rows):
    with open(os.path.join(packed, "meta.json")) as handle:
        meta = json.load(handle)
    count = meta["shards"][shard]
    tok = np.memmap(os.path.join(packed, f"shard-{shard:05d}.bin"), dtype=np.uint32, mode="r", shape=(count, meta["seq_len"]))
    stop = min(row0 + rows, count)
    return torch.from_numpy(np.asarray(tok[row0:stop]).astype(np.int64))


def loss_on(model, trainer, batch_all):
    weight = model.lm_head.weight.float()
    total = 0.0
    n = 0
    for start in range(0, batch_all.shape[0], 4):
        batch = batch_all[start:start + 4].cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state[:, :-1, :]
        with torch.inference_mode():
            ce = trainer.cross_entropy(hidden.float(), weight, batch[:, 1:], 1024)
        total += ce.item() * batch[:, 1:].numel()
        n += batch[:, 1:].numel()
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--packed", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--shards", type=int, nargs="+", required=True)
    parser.add_argument("--row0", type=int, default=4000)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    trainer = train_kd()
    sets = {shard: rows_from(args.packed, shard, args.row0, args.rows) for shard in args.shards}
    result = {"run_dir": args.run_dir, "packed": args.packed, "row0": args.row0, "rows": args.rows, "loss": {}}
    for step in args.steps:
        path = os.path.join(args.run_dir, f"step_{step:07d}.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = trainer.build_student(trainer.config_for_checkpoint(state), for_training=False)
        model.load_state_dict(state["model"])
        del state
        gc.collect()
        model.to(device="cuda", dtype=torch.bfloat16).eval()
        result["loss"][step] = {shard: round(loss_on(model, trainer, batch), 4) for shard, batch in sets.items()}
        print(f"step {step}", json.dumps(result["loss"][step]), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
