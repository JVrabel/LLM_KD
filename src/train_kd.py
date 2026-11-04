#!/usr/bin/env python3
"""Train a from-scratch Qwen3 student on packed token shards.

The 0.6B pilot fits on one GPU, so this is a single-process trainer. It is the
run that has to be right before any multi-GPU work: the vocabulary loss is
computed in chunks over the 248k vocabulary, and checkpoints restore the model,
optimizer, schedule, RNG and data cursor.

``kd_ratio`` 0 is next-token cross-entropy. ``kd_ratio`` 1 is pure forward KL
against the teacher's stored top-64 log probabilities, with the student's
probabilities taken from the full vocabulary. A value in between mixes the two.

Data is ``data_dir`` (plus ``logit_dir``) or a ``data`` list of such sources.
``training.shuffle: true`` draws sequences in a seeded random order across all
sources; without it they come in pack order, as in the October 2026 pilots.

Launch: python src/train_kd.py --config config/pilot_0.6b_smoke.yaml
"""

import argparse
import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

_SHARED = dict(
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
# Shapes follow the dense Qwen3 models. The vocabulary override makes the
# measured sizes about 0.70B, 1.92B, and 4.27B parameters.
STUDENTS = {
    "0.6b": dict(
        hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128, **_SHARED,
    ),
    "1.7b": dict(
        hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128, **_SHARED,
    ),
    "4b": dict(
        hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, **_SHARED,
    ),
}
STUDENT_0_6B = STUDENTS["0.6b"]


def build_student(config, for_training=True):
    model = Qwen3ForCausalLM(Qwen3Config(**config))
    model.config.use_cache = not for_training
    if for_training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def config_for_checkpoint(state):
    """Architecture stored in a new checkpoint, or the legacy 0.6B shape."""
    saved = state.get("model_config")
    if saved is not None:
        return saved
    return STUDENTS["0.6b"]


def resolve_student(cfg, state=None):
    size = str(cfg.get("model_size", "0.6b"))
    if size not in STUDENTS:
        raise SystemExit(f"unknown model_size {size}; choose from {', '.join(STUDENTS)}")
    config = STUDENTS[size]
    if state is not None:
        saved = state.get("model_config")
        saved_size = state.get("model_size")
        if saved is not None and saved != config:
            raise SystemExit(f"checkpoint architecture does not match model_size {size}")
        if saved_size is not None and saved_size != size:
            raise SystemExit(f"checkpoint model_size is {saved_size}, config says {size}")
        if saved is None and size != "0.6b":
            raise SystemExit("this checkpoint has no architecture metadata and is a 0.6B run")
    return size, config


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


def objective_name(kd_ratio):
    if kd_ratio <= 0.0:
        return "next_token"
    if kd_ratio >= 1.0:
        return "forward_kl"
    return "mixed"


def topk_forward_kl(hidden, weight, teacher_ids, teacher_logprobs, chunk):
    """Mean forward KL from the teacher's top-k, plus one bin for the leftover mass.

    Stored values are log probabilities under the full vocabulary. Sentinel
    slots (about -1e4) are ignored. The k stored tokens and the single
    leftover bin ``1 - sum(p)`` form the teacher distribution. Student
    probabilities come from a softmax over the whole vocabulary, and the
    leftover bin is whatever mass the student placed outside those k tokens.
    """
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_ids = teacher_ids.reshape(-1, teacher_ids.shape[-1]).to(dtype=torch.long)
    flat_lp = teacher_logprobs.reshape(-1, teacher_logprobs.shape[-1]).float()
    valid = flat_lp > -1.0e3
    safe_ids = flat_ids.clamp(min=0, max=weight.shape[0] - 1)
    total = hidden.new_zeros((), dtype=torch.float32)
    positions = flat_hidden.shape[0]
    for start in range(0, positions, chunk):
        stop = start + chunk
        logits = F.linear(flat_hidden[start:stop], weight).float()
        student_log_full = F.log_softmax(logits, dim=-1)
        ids = safe_ids[start:stop]
        lp = flat_lp[start:stop]
        mask = valid[start:stop]
        student_log = student_log_full.gather(1, ids)
        teacher_p = torch.where(mask, lp.exp(), torch.zeros_like(lp))
        teacher_log = torch.where(mask, lp, torch.zeros_like(lp))
        head = torch.where(
            mask, teacher_p * (teacher_log - student_log), torch.zeros_like(teacher_p),
        ).sum(dim=-1)
        p_rest = (1.0 - teacher_p.sum(dim=-1)).clamp(min=0.0)
        q_support = torch.where(mask, student_log.exp(), torch.zeros_like(student_log)).sum(dim=-1)
        q_rest = (1.0 - q_support).clamp(min=1e-8)
        rest_log = torch.where(p_rest > 0, p_rest.log(), torch.zeros_like(p_rest))
        rest = p_rest * (rest_log - q_rest.log())
        total = total + (head + rest).sum()
    return total / positions


def data_sources(cfg):
    """Packed sources from ``data`` (a list) or the single ``data_dir``/``logit_dir`` pair.

    Each entry has ``dir``, an optional ``logit_dir`` and an optional ``shards``
    cap, which uses only the first that many shards of the pack (the Apollo
    pack has 33 shards but teacher logits for 30).
    """
    entries = cfg.get("data")
    if entries is None:
        entries = [{"dir": cfg["data_dir"], "logit_dir": cfg.get("logit_dir")}]
    sources = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"dir": entry}
        sources.append({
            "dir": entry["dir"],
            "logit_dir": entry.get("logit_dir"),
            "shards": entry.get("shards"),
        })
    return sources


class ShardCursor:
    """Batches of packed sequences, in file order or in a seeded random order.

    Without ``shuffle_seed`` the sequences come shard by shard, source by
    source, which is how the October 2026 pilots ran; the packs are in corpus
    order, so that run saw one slice of the corpus after another and forgot
    the early ones. With ``shuffle_seed`` every epoch is a fresh permutation
    of all sequences across all sources. The stored teacher logits stay
    aligned because sequences are permuted, not repacked. The state is the
    position in the current epoch's order; a legacy ``shard``/``index`` state
    is converted on load.
    """

    def __init__(self, sources, batch_seqs, state, use_logits, shuffle_seed=None):
        self.batch = batch_seqs
        self.shuffle_seed = shuffle_seed
        self.seq_len = None
        self.paths, self.counts, self.logit_stems = [], [], []
        for source in sources:
            with open(os.path.join(source["dir"], "meta.json")) as handle:
                meta = json.load(handle)
            if self.seq_len is None:
                self.seq_len = meta["seq_len"]
            elif meta["seq_len"] != self.seq_len:
                raise SystemExit(f"{source['dir']} has seq_len {meta['seq_len']}, expected {self.seq_len}")
            counts = meta["shards"]
            if source.get("shards") is not None:
                counts = counts[: source["shards"]]
            logit_dir = source.get("logit_dir") if use_logits else None
            if use_logits and not logit_dir:
                raise SystemExit(f"{source['dir']} has no logit_dir but the objective needs teacher logits")
            for number, count in enumerate(counts):
                self.paths.append(os.path.join(source["dir"], f"shard-{number:05d}.bin"))
                self.counts.append(count)
                stem = os.path.join(logit_dir, f"shard-{number:05d}") if logit_dir else None
                if stem is not None and not os.path.exists(stem + ".done"):
                    raise RuntimeError(f"teacher logit shard is incomplete: {stem}.done")
                self.logit_stems.append(stem)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)
        self.total = int(self.starts[-1])
        self.epoch = int(state.get("epoch", 0))
        if "position" in state:
            self.position = int(state["position"])
        else:
            self.position = int(self.starts[state.get("shard", 0)]) + int(state.get("index", 0))
        self.order = self._order()
        self._open = {}

    def _order(self):
        if self.shuffle_seed is None:
            return np.arange(self.total)
        return np.random.default_rng([self.shuffle_seed, self.epoch]).permutation(self.total)

    def _shard(self, number):
        if number not in self._open:
            if self.shuffle_seed is None:
                self._open.clear()
            tokens = np.memmap(
                self.paths[number], dtype=np.uint32, mode="r", shape=(self.counts[number], self.seq_len),
            )
            ids = lp = None
            stem = self.logit_stems[number]
            if stem is not None:
                ids = np.load(stem + ".ids.npy", mmap_mode="r")
                lp = np.load(stem + ".lp.npy", mmap_mode="r")
            self._open[number] = (tokens, ids, lp)
        return self._open[number]

    def describe(self):
        mode = "file order" if self.shuffle_seed is None else f"shuffled with seed {self.shuffle_seed}"
        return f"{self.total} sequences in {len(self.paths)} shards, {mode}, epoch {self.epoch} position {self.position}"

    def state(self):
        return {"position": self.position, "epoch": self.epoch, "shuffle_seed": self.shuffle_seed}

    def next_batch(self):
        rows, teacher_ids, teacher_lp = [], [], []
        while len(rows) < self.batch:
            if self.position >= self.total:
                self.position = 0
                self.epoch += 1
                self.order = self._order()
            global_index = int(self.order[self.position])
            number = int(np.searchsorted(self.starts, global_index, side="right") - 1)
            row = global_index - int(self.starts[number])
            tokens, ids, lp = self._shard(number)
            rows.append(np.array(tokens[row], copy=True))
            if ids is not None:
                # Position t holds the distribution of token t; the student
                # hidden state at t-1 predicts it, so the first position is unused.
                teacher_ids.append(np.array(ids[row, 1:], copy=True))
                teacher_lp.append(np.array(lp[row, 1:], copy=True))
            self.position += 1
        batch = torch.from_numpy(np.stack(rows).astype(np.int64))
        if not teacher_ids:
            return batch, None, None
        return (
            batch,
            torch.from_numpy(np.stack(teacher_ids).astype(np.int64)),
            torch.from_numpy(np.stack(teacher_lp).astype(np.float32)),
        )


def learning_rate(step, total, warmup, decay_start, base, floor, rewarmup=None):
    """Warm-up, constant, cosine decay to ``floor * base``.

    ``rewarmup`` is an optional ``(from_step, steps)`` pair for a run continued
    from an annealed checkpoint: the rate ramps linearly from ``floor * base``
    back to the schedule's value over ``steps`` steps starting at ``from_step``.
    """
    if step < warmup:
        lr = base * (step + 1) / max(1, warmup)
    elif step < decay_start:
        lr = base
    else:
        progress = (step - decay_start) / max(1, total - decay_start)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = base * (floor + (1 - floor) * cosine)
    if rewarmup and rewarmup[0] <= step < rewarmup[0] + rewarmup[1]:
        fraction = (step - rewarmup[0] + 1) / rewarmup[1]
        lr = lr * (floor + (1 - floor) * fraction)
    return lr


def load_latest_checkpoint(output_dir):
    """The newest step_*.pt in output_dir that loads, as (path, state).

    A checkpoint cut off by a wall-clock kill fails to load and is skipped
    with a message. (None, None) when the directory has no usable checkpoint.
    """
    for path in sorted(glob.glob(os.path.join(output_dir, "step_*.pt")), reverse=True):
        try:
            return path, torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - any unreadable file is skipped
            print(f"skipping unreadable checkpoint {path}: {type(error).__name__}: {error}", flush=True)
    return None, None


def save_checkpoint(path, model, optimizer, step, cursor, tokens, model_size, model_config, objective):
    torch.save({
        "checkpoint_version": 2,
        "objective": objective,
        "model_size": model_size,
        "model_config": model_config,
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


def restore_checkpoint(state, model, optimizer):
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume-latest", action="store_true",
        help="when the config has no resume key, continue from the newest loadable "
             "step_*.pt in output_dir, or start fresh if there is none; lets a run "
             "longer than one allocation be chained as several jobs",
    )
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda"
    # --resume-latest prefers the newest checkpoint already in output_dir, so a
    # chained continuation run (config `resume:` pointing at another run's
    # checkpoint) picks up its own progress on the second job instead of
    # restarting from the foreign checkpoint.
    resume, state = None, None
    if args.resume_latest:
        resume, state = load_latest_checkpoint(cfg["output_dir"])
        print(f"resume-latest: {resume or 'no checkpoint in output_dir'}", flush=True)
    if state is None and cfg.get("resume"):
        resume = cfg["resume"]
        state = torch.load(resume, map_location="cpu", weights_only=False)
    model_size, model_config = resolve_student(cfg, state)
    model = build_student(model_config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"student {model_size} parameters {parameters/1e9:.3f}B", flush=True)

    train = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train["learning_rate"], betas=(0.9, 0.95),
        weight_decay=0.1, fused=True,
    )
    cursor_state = {"position": 0, "epoch": 0}
    step = 0
    tokens = 0
    if state is not None:
        restore_checkpoint(state, model, optimizer)
        step = state["step"]
        tokens = state["tokens"]
        cursor_state = state["cursor"]
        del state
        print(f"resumed at step {step} from {resume}", flush=True)

    kd_ratio = float(train.get("kd_ratio", 0.0))
    if kd_ratio < 0.0 or kd_ratio > 1.0:
        raise SystemExit(f"kd_ratio must be between 0 and 1, got {kd_ratio}")
    shuffle_seed = seed if train.get("shuffle", False) else None
    if resume and cursor_state.get("shuffle_seed") != shuffle_seed:
        raise SystemExit(
            f"checkpoint cursor has shuffle_seed {cursor_state.get('shuffle_seed')}, "
            f"config gives {shuffle_seed}; the data order would change mid-run"
        )
    cursor = ShardCursor(
        data_sources(cfg), train["batch_sequences"], cursor_state,
        use_logits=kd_ratio > 0.0, shuffle_seed=shuffle_seed,
    )
    print(f"data {cursor.describe()}", flush=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    total = train["steps"]
    warmup = train["warmup_steps"]
    decay_start = train["decay_start_step"]
    base_lr = train["learning_rate"]
    temperature = float(train.get("temperature", 1.0))
    if kd_ratio > 0.0 and temperature != 1.0:
        raise SystemExit(
            "distillation uses the stored full-vocabulary log probabilities, "
            f"which are already at temperature 1; config has temperature {temperature}"
        )
    objective = objective_name(kd_ratio)
    rewarmup = None
    if train.get("rewarmup_steps"):
        rewarmup = (int(train["rewarmup_from_step"]), int(train["rewarmup_steps"]))
        print(f"rewarmup over {rewarmup[1]} steps from step {rewarmup[0]}", flush=True)
    chunk = train.get("logit_chunk", 1024)
    accum = train.get("grad_accumulation_steps", 1)
    print(f"objective {objective} kd_ratio {kd_ratio:g}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < total:
        lr = learning_rate(step, total, warmup, decay_start, base_lr, train.get("lr_floor", 0.1), rewarmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        ce_sum = kd_sum = loss_sum = 0.0
        step_tokens = 0
        will_log = (step + 1) % train["log_every"] == 0 or step == 0
        for _ in range(accum):
            batch, teacher_ids, teacher_lp = cursor.next_batch()
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state[:, :-1, :]
            labels = batch[:, 1:]
            ce_loss = None
            if kd_ratio < 1.0:
                ce_loss = cross_entropy(hidden, model.lm_head.weight, labels, chunk)
            kd_loss = None
            if kd_ratio > 0.0:
                if teacher_ids is None:
                    raise SystemExit(f"kd_ratio is {kd_ratio} but the data has no teacher logits")
                kd_loss = topk_forward_kl(
                    hidden, model.lm_head.weight,
                    teacher_ids.to(device), teacher_lp.to(device), chunk,
                )
            if kd_ratio <= 0.0:
                loss = ce_loss
            elif kd_ratio >= 1.0:
                loss = kd_loss
            else:
                loss = (1.0 - kd_ratio) * ce_loss + kd_ratio * kd_loss
            (loss / accum).backward()
            if ce_loss is None and will_log:
                with torch.no_grad():
                    ce_loss = cross_entropy(hidden, model.lm_head.weight, labels, chunk)
            loss_sum += loss.item()
            if ce_loss is not None:
                ce_sum += ce_loss.item()
            if kd_loss is not None:
                kd_sum += kd_loss.item()
            step_tokens += batch.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens += step_tokens
        if step % train["log_every"] == 0 or step == 1:
            components = ""
            if kd_ratio > 0.0:
                components += f" kd {kd_sum / accum:.4f}"
            components += f" ce {ce_sum / accum:.4f}"
            print(
                f"step {step} loss {loss_sum / accum:.4f}{components} "
                f"lr {lr:.6e} tokens {tokens}",
                flush=True,
            )
        if step % train["save_every"] == 0 or step == total:
            path = os.path.join(cfg["output_dir"], f"step_{step:07d}.pt")
            save_checkpoint(
                path, model, optimizer, step, cursor, tokens,
                model_size, model_config, objective,
            )
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
