#!/usr/bin/env python3
"""Smoke test: Qwen3.5-35B-A3B-Base under vLLM on one GPU.

Loads the base model in text-only mode, continues one medical and one code
prompt, and reports perplexity plus tokens/second on a few dozen documents.
"""

import gzip
import json
import math
import re
import time

from vllm import LLM, SamplingParams

MODEL = "/projects/u6wc/savants/models/Qwen3.5-35B-A3B-Base"
PYTHON_JSONL = "/projects/u6wc/savants/data/raw/python/python_clean.jsonl"
PUBMED_XML = "/projects/u6wc/savants/data/raw/pubmed_baseline/pubmed26n0001.xml.gz"

N_DOCS = 24
MIN_CHARS = 800
MAX_CHARS = 6000


def python_docs():
    docs = []
    with open(PYTHON_JSONL) as handle:
        for line in handle:
            text = json.loads(line)["text"].strip()
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                docs.append(text)
            if len(docs) == N_DOCS:
                break
    return docs


def pubmed_docs():
    docs = []
    with gzip.open(PUBMED_XML, "rt", errors="replace") as handle:
        chunk = []
        for line in handle:
            chunk.append(line)
            if "</PubmedArticle>" in line:
                article = "".join(chunk)
                chunk = []
                for match in re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.S):
                    text = re.sub(r"<[^>]+>", " ", match)
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) >= MIN_CHARS:
                        docs.append(text[:MAX_CHARS])
                if len(docs) >= N_DOCS:
                    break
    return docs[:N_DOCS]


def perplexity(outputs):
    nlls = []
    for output in outputs:
        for index, entry in enumerate(output.prompt_logprobs):
            if not entry:
                continue
            token_id = output.prompt_token_ids[index]
            nlls.append(-entry[token_id].logprob)
    mean = sum(nlls) / len(nlls)
    return math.exp(mean), len(nlls), mean


def main():
    print("loading model", flush=True)
    started = time.time()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        language_model_only=True,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    sets = {"python": python_docs(), "pubmed": pubmed_docs()}
    for name, docs in sets.items():
        print(f"{name}: {len(docs)} documents", flush=True)

    continuation = SamplingParams(temperature=0, max_tokens=40)
    for name in ("pubmed", "python"):
        result = llm.generate([sets[name][0][:1500]], continuation)[0]
        print(f"--- continuation ({name}) ---", flush=True)
        print(result.outputs[0].text[:500], flush=True)

    scoring = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    for name, docs in sets.items():
        started = time.time()
        outputs = llm.generate(docs, scoring)
        elapsed = time.time() - started
        ppl, n_tokens, mean_nll = perplexity(outputs)
        print(
            f"{name}: perplexity {ppl:.2f}  nll {mean_nll:.3f}  "
            f"tokens {n_tokens}  {n_tokens / elapsed:.0f} tok/s  ({elapsed:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
