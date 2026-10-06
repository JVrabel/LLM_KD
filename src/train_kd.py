#!/usr/bin/env python3
"""Train a from-scratch Qwen3 student on packed token shards.

The 0.6B pilot fits on one GPU, so this is a single-process trainer. It is the
run that has to be right before any multi-GPU work: cross-entropy is computed
in chunks over the 248k vocabulary, checkpoints restore the model, optimizer,
schedule, RNG and data cursor, and an optional top-64 teacher logit file adds
a forward-KL term.

Launch: python src/train_kd.py --config config/pilot_0.6b_smoke.yaml
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

STUDENT_0_6B = dict(
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000,
    tie_word_embeddings=True,
    attention_bias=False,
    vocab_size=248320,
    eos_token_id=248044,
    bos_token_id=248044,
    pad_token_id=248044,
)


def build_student():
    model = Qwen3ForCausalLM(Qwen3Config(**STUDENT_0_6B))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def cross_entropy(hidden, weight, labels, chunk):
    """Mean next-token loss. ``hidden`` predicts ``labels`` and never materializes the full vocabulary."""
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_hidden.shape[0], chunk):
        stop = start + chunk
        logits = F.linear(flat_hidden[start:stop], weight).float()
        total = total + F.cross_entropy(logits, flat_labels[start:stop], reduction="sum")
    return total / flat_labels.numel()


def topk_kl(hidden, weight, teacher_ids, teacher_logprobs, temperature, chunk):
    """Forward KL between the teacher's renormalized top-k and the student on those ids."""
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_ids = teacher_ids.reshape(-1, teacher_ids.shape[-1])
    flat_logprobs = teacher_logprobs.reshape(-1, teacher_logprobs.shape[-1])
    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, flat_hidden.shape[0], chunk):
        stop = start + chunk
        chosen = weight[flat_ids[start:stop]]
        logits = (flat_hidden[start:stop].unsqueeze(1) * chosen).sum(-1).float()
        student_log = F.log_softmax(logits / temperature, dim=-1)
        teacher = F.softmax(flat_logprobs[start:stop].float() / temperature, dim=-1)
        total = total + F.kl_div(student_log, teacher, reduction="sum")
        count += stop - start
    return total / count * temperature ** 2


class ShardCursor:
    def __init__(self, data_dir, batch_seqs, shard, index, epoch, logit_dir=None):
        with open(os.path.join(data_dir, "meta.json")) as handle:
            meta = json.load(handle)
        self.seq_len = meta["seq_len"]
        self.batch = batch_seqs
        self.paths = []
        self.counts = []
        self.logit_dir = logit_dir
        for number, count in enumerate(meta["shards"]):
            self.paths.append(os.path.join(data_dir, f"shard-{number:05d}.bin"))
            self.counts.append(count)
        self.shard = shard
        self.index = index
        self.epoch = epoch
        self._map = self._ids = self._lp = None
        self._open()

    def _open(self):
        count = self.counts[self.shard]
        self._map = np.memmap(
            self.paths[self.shard], dtype=np.uint32, mode="r", shape=(count, self.seq_len),
        )
        self._ids = self._lp = None
        if self.logit_dir:
            stem = f"shard-{self.shard:05d}"
            ids_path = os.path.join(self.logit_dir, stem + ".ids.npy")
            self._ids = np.load(ids_path, mmap_mode="r")
            self._lp = np.load(os.path.join(self.logit_dir, stem + ".lp.npy"), mmap_mode="r")

    def state(self):
        return {"shard": self.shard, "index": self.index, "epoch": self.epoch}

    def next_batch(self):
        rows, teacher_ids, teacher_lp = [], [], []
        while len(rows) < self.batch:
            if self.index >= self.counts[self.shard]:
                self.shard += 1
                self.index = 0
                if self.shard >= len(self.paths):
                    self.shard = 0
                    self.epoch += 1
                self._open()
            rows.append(np.array(self._map[self.index], copy=True))
            if self._ids is not None:
                # Position t holds the distribution of token t; the student
                # hidden state at t-1 predicts it, so the first position is unused.
                teacher_ids.append(np.array(self._ids[self.index, 1:], copy=True))
                teacher_lp.append(np.array(self._lp[self.index, 1:], copy=True))
            self.index += 1
        batch = torch.from_numpy(np.stack(rows).astype(np.int64))
        if not teacher_ids:
            return batch, None, None
        return (
            batch,
            torch.from_numpy(np.stack(teacher_ids).astype(np.int64)),
            torch.from_numpy(np.stack(teacher_lp).astype(np.float32)),
        )


def learning_rate(step, total, warmup, decay_start, base, floor):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    if step < decay_start:
        return base
    progress = (step - decay_start) / max(1, total - decay_start)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base * (floor + (1 - floor) * cosine)


def save_checkpoint(path, model, optimizer, step, cursor, tokens):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens": tokens,
        "cursor": cursor.state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, path)


def load_checkpoint(path, model, optimizer):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda"
    model = build_student().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"student parameters {parameters/1e9:.3f}B", flush=True)

    train = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train["learning_rate"], betas=(0.9, 0.95),
        weight_decay=0.1, fused=True,
    )
    cursor_state = {"shard": 0, "index": 0, "epoch": 0}
    step = 0
    tokens = 0
    resume = cfg.get("resume")
    if resume:
        state = load_checkpoint(resume, model, optimizer)
        step = state["step"]
        tokens = state["tokens"]
        cursor_state = state["cursor"]
        print(f"resumed at step {step}", flush=True)

    logit_dir = cfg.get("logit_dir") if train.get("kd_ratio", 0.0) else None
    cursor = ShardCursor(
        cfg["data_dir"], train["batch_sequences"],
        cursor_state["shard"], cursor_state["index"], cursor_state["epoch"],
        logit_dir=logit_dir,
    )
    os.makedirs(cfg["output_dir"], exist_ok=True)
    total = train["steps"]
    warmup = train["warmup_steps"]
    decay_start = train["decay_start_step"]
    base_lr = train["learning_rate"]
    kd_ratio = train.get("kd_ratio", 0.0)
    temperature = train.get("temperature", 1.0)
    chunk = train.get("logit_chunk", 1024)

    model.train()
    while step < total:
        batch, teacher_ids, teacher_lp = cursor.next_batch()
        batch = batch.to(device, non_blocking=True)
        lr = learning_rate(step, total, warmup, decay_start, base_lr, train.get("lr_floor", 0.1))
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state[:, :-1, :]
        labels = batch[:, 1:]
        loss = cross_entropy(hidden, model.lm_head.weight, labels, chunk)
        if kd_ratio:
            if teacher_ids is None:
                raise SystemExit(f"kd_ratio is {kd_ratio} but {logit_dir} has no logit shard")
            loss = loss + kd_ratio * topk_kl(
                hidden, model.lm_head.weight,
                teacher_ids.to(device), teacher_lp.to(device), temperature, chunk,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens += batch.numel()
        if step % train["log_every"] == 0 or step == 1:
            print(f"step {step} loss {loss.item():.4f} lr {lr:.6e} tokens {tokens}", flush=True)
        if step % train["save_every"] == 0 or step == total:
            path = os.path.join(cfg["output_dir"], f"step_{step:07d}.pt")
            save_checkpoint(path, model, optimizer, step, cursor, tokens)
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
