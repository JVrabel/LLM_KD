#!/usr/bin/env python3
"""Keep the Apollo half of the tagged medical JSONL.

The full medical file is mostly PubMed abstracts. Apollo is the textbook-style
subset used as the quality arm. Records are already tagged with ``source``.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        print(f"already filtered: {args.out}", flush=True)
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".partial"
    seen = kept = 0
    with open(args.src) as source, open(tmp, "w") as dest:
        for line in source:
            if not line.strip():
                continue
            seen += 1
            row = json.loads(line)
            if row.get("source") != "apollo":
                continue
            dest.write(json.dumps({"text": row["text"], "source": "apollo"}) + "\n")
            kept += 1
            if kept % 100000 == 0:
                print(f"kept {kept}  seen {seen}", flush=True)
    os.rename(tmp, args.out)
    print(f"DONE kept {kept} seen {seen}", flush=True)


if __name__ == "__main__":
    main()
