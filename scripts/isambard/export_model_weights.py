"""Model-only bf16 copies of training checkpoints for the end-of-project
backup (22 Nov 2026). The full checkpoints carry fp32 weights plus Adam
state (8.3 GB for 0.6B, 23 GB for 1.7B); the export keeps the weights in
bf16 and the small metadata (model_size, model_config, step, tokens,
objective), so scripts/student_mmlu.py can load the result unchanged.
Runs on the login node with the training venv: no GPU, mmap load. The login
node kills processes above a few GB, so the weights are written as parts of
at most --max-part-gb each into a directory <run>_<step>_model_bf16/ with
meta.pt plus part-NN.pt; student_mmlu.load_model merges them.
  python export_model_weights.py --out backup runs/*/step_0091551.pt ...
"""
import argparse
import gc
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--out", required=True)
parser.add_argument("--max-part-gb", type=float, default=1.0)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
KEEP = ("checkpoint_version", "objective", "model_size", "model_config", "step", "tokens")
limit = int(args.max_part_gb * 1e9)
for path in args.checkpoints:
    run = os.path.basename(os.path.dirname(path))
    step = os.path.basename(path).replace(".pt", "")
    dst = os.path.join(args.out, f"{run}_{step}_model_bf16")
    if os.path.exists(os.path.join(dst, "meta.pt")):
        print("skip", dst, flush=True)
        continue
    os.makedirs(dst, exist_ok=True)
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    meta = {k: state[k] for k in KEEP if k in state}
    part, part_bytes, n_parts, n_params = {}, 0, 0, 0
    def flush():
        global part, part_bytes, n_parts
        if not part:
            return
        torch.save(part, os.path.join(dst, f"part-{n_parts:02d}.pt"))
        n_parts += 1
        part, part_bytes = {}, 0
        gc.collect()
    for k, v in state["model"].items():
        t = v.to(torch.bfloat16) if v.is_floating_point() else v.clone()
        part[k] = t
        part_bytes += t.numel() * t.element_size()
        n_params += t.numel()
        if part_bytes >= limit:
            flush()
    flush()
    meta["parts"] = n_parts
    torch.save(meta, os.path.join(dst, "meta.pt"))
    size = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)) / 1e9
    print(f"wrote {dst} {n_params/1e9:.2f}B params {size:.1f} GB in {n_parts} parts", flush=True)
    del state
    gc.collect()
print("DONE", flush=True)
