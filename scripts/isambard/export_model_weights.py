"""Model-only bf16 copies of training checkpoints for the end-of-project
backup (22 Nov 2026). The full checkpoints carry fp32 weights plus Adam
state (8.3 GB for 0.6B, 23 GB for 1.7B); the export keeps the weights in
bf16 and the small metadata (model_size, model_config, step, tokens,
objective), so scripts/student_mmlu.py can load the result unchanged.
Runs on the login node with the training venv: no GPU, mmap load.
  python export_model_weights.py --out backup runs/*/step_0091551.pt ...
"""
import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--out", required=True)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
KEEP = ("checkpoint_version", "objective", "model_size", "model_config", "step", "tokens")
for path in args.checkpoints:
    run = os.path.basename(os.path.dirname(path))
    step = os.path.basename(path).replace(".pt", "")
    dst = os.path.join(args.out, f"{run}_{step}_model_bf16.pt")
    if os.path.exists(dst):
        print("skip", dst, flush=True)
        continue
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    model = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v.clone())
        for k, v in state["model"].items()
    }
    out = {k: state[k] for k in KEEP if k in state}
    out["model"] = model
    torch.save(out, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    params = sum(v.numel() for v in model.values()) / 1e9
    print(f"wrote {dst} {params:.2f}B params {os.path.getsize(dst)/1e9:.1f} GB", flush=True)
print("DONE", flush=True)
