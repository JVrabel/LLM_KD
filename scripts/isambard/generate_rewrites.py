#!/usr/bin/env python3
"""Teacher-written rewrites of Apollo documents (20 Nov 2026, "denser knowledge").

Each source document is turned into one or more rewrites by the base teacher
with a plain-text template (no chat format): a textbook passage, exam-style
questions with answers, or study notes. Output is JSONL ``{"text": ..., "doc":
line index, "style": name}``, ready for ``pack_shards.py``; teacher logits are
then scored on the packed shards with the usual ``precompute_logits.py`` so the
training pipeline is unchanged.

Parallel instances take disjoint line ranges (``--start`` / ``--count``). The
output file is appended and resumed by counting finished (doc, style) pairs.
"""

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams

STYLES = {
    "textbook": (
        "a rewrite of the excerpt as a self-contained section of a medical textbook that "
        "defines the key terms, explains the mechanisms and states every clinical fact "
        "explicitly, without referring to the excerpt",
        "Textbook section",
    ),
    "questions": (
        "five exam-style multiple-choice questions testing the medical knowledge in the "
        "excerpt, each with four options (A-D), the correct answer, and a short explanation "
        "that states the underlying fact itself and never refers to the excerpt or the text",
        "Questions",
    ),
    "notes": (
        "concise study notes: a numbered list of the medical facts from the excerpt that a "
        "student must remember, each written as a complete, self-contained sentence that "
        "names its subject and never refers to the excerpt",
        "Study notes",
    ),
}
TEMPLATE = (
    "The following is an excerpt from a medical text, followed by {task}.\n\n"
    "### Excerpt\n{doc}\n\n### {heading}\n"
)


def read_docs(path, start, count, tokenizer, min_tokens, max_tokens):
    """Yield (line index, text) for non-blank lines in [start, start+count)."""
    with open(path, "rb") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if index >= start + count:
                break
            if not line.strip():
                continue
            text = json.loads(line)["text"].strip()
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < min_tokens:
                continue
            if len(ids) > max_tokens:
                text = tokenizer.decode(ids[:max_tokens])
            yield index, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0, help="first corpus line")
    parser.add_argument("--count", type=int, required=True, help="number of corpus lines")
    parser.add_argument("--styles", default="textbook,questions,notes")
    parser.add_argument("--batch", type=int, default=1024, help="prompts per generate call")
    parser.add_argument("--max-new", type=int, default=1024)
    parser.add_argument("--min-doc-tokens", type=int, default=120)
    parser.add_argument("--max-doc-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    styles = [s for s in args.styles.split(",") if s]
    for s in styles:
        if s not in STYLES:
            raise SystemExit(f"unknown style {s}")

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    done.add((row["doc"], row["style"]))
                except (ValueError, KeyError):
                    pass
        print(f"resume: {len(done)} rewrites already in {args.out}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_doc_tokens + args.max_new + 256,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        language_model_only=True,
        trust_remote_code=True,
        enforce_eager=True,  # the compiled/cudagraph path hung TP workers after a few batches (20 Nov 2026)
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature, top_p=0.95, max_tokens=args.max_new,
        stop=["\n### ", "\n\n\n\n"], repetition_penalty=1.05,
    )

    pending = []
    started = time.perf_counter()
    generated = 0
    written = 0

    def run(batch):
        nonlocal generated, written
        prompts = [TEMPLATE.format(task=STYLES[s][0], heading=STYLES[s][1], doc=text)
                   for _, text, s in batch]
        results = llm.generate(prompts, sampling, use_tqdm=False)
        with open(args.out, "a") as handle:
            for (index, _, s), result in zip(batch, results):
                out = result.outputs[0]
                generated += len(out.token_ids)
                text = out.text.strip()
                if len(out.token_ids) < 64 or out.finish_reason == "length" and len(out.token_ids) >= args.max_new:
                    # too short, or ran away without finishing: keep only clean outputs
                    if out.finish_reason == "length":
                        text = text.rsplit("\n", 1)[0].strip()  # drop the cut-off line
                    if len(out.token_ids) < 64:
                        continue
                handle.write(json.dumps({"text": f"{STYLES[s][1]}\n{text}", "doc": index, "style": s}) + "\n")
                written += 1
        elapsed = time.perf_counter() - started
        print(f"written {written} rewrites, {generated} new tokens, {generated / elapsed:.0f} tok/s", flush=True)

    for index, text in read_docs(args.corpus, args.start, args.count, tokenizer,
                                 args.min_doc_tokens, args.max_doc_tokens):
        for s in styles:
            if (index, s) in done:
                continue
            pending.append((index, text, s))
            if len(pending) >= args.batch:
                run(pending)
                pending = []
    if pending:
        run(pending)
    print("DONE", json.dumps({"out": args.out, "start": args.start, "count": args.count,
                              "written": written, "generated_tokens": generated}), flush=True)


if __name__ == "__main__":
    main()
