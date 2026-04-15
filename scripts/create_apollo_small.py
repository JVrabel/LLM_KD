from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic small Apollo subset.")
    parser.add_argument("--input", required=True, help="Path to the source JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory where the subset should be written.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional fraction of rows to keep.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=128,
        help="Target number of rows to keep. Default: 128.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    source_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = count_lines(source_path)
    if args.fraction is not None:
        sample_size = max(1, int(round(total_rows * args.fraction)))
    else:
        sample_size = max(1, min(args.target_rows, total_rows))

    rng = random.Random(args.seed)
    sampled_indices = set(rng.sample(range(total_rows), sample_size))
    output_path = output_dir / source_path.name

    kept_rows = 0
    with source_path.open("r", encoding="utf-8") as source_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for idx, line in enumerate(source_handle):
            if idx in sampled_indices:
                output_handle.write(line)
                kept_rows += 1

    metadata = {
        "source": str(source_path),
        "output": str(output_path),
        "fraction": args.fraction,
        "target_rows": args.target_rows,
        "seed": args.seed,
        "total_rows": total_rows,
        "sampled_rows": kept_rows,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
