#!/usr/bin/env python3
"""Download ApolloCorpus SFT splits from Hugging Face and merge to JSONL.

Source dataset:
  https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus/tree/main/train/sft

Each SFT file is a JSON array. Records are typically [question/prompt, answer] lists,
which matches the instruction formats supported by src/data_setup.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download

REPO_ID = "FreedomIntelligence/ApolloCorpus"
REPO_TYPE = "dataset"
SFT_PREFIX = "train/sft"

SUBSETS: dict[str, list[str]] = {
    # English medical instruction data for Phase 2 tuning.
    "medical_en": [
        "medicalExam_en_clean.json",
        "medicalPatient_en.json",
    ],
    "medical_en_all": [
        "medicalExam_en.json",
        "medicalExam_en_clean.json",
        "medicalPatient_en.json",
    ],
}


def download_file(filename: str, cache_dir: Path | None) -> Path:
    print(f"Downloading {filename} ...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{SFT_PREFIX}/{filename}",
        repo_type=REPO_TYPE,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(downloaded)


def json_array_to_jsonl(source_path: Path, output_path: Path) -> int:
    with source_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {source_path}, got {type(records)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def merge_jsonl_files(inputs: Iterable[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with output_path.open("w", encoding="utf-8") as out_handle:
        for input_path in inputs:
            with input_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if line.strip():
                        out_handle.write(line if line.endswith("\n") else line + "\n")
                        total_rows += 1
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ApolloCorpus SFT data and build JSONL.")
    parser.add_argument(
        "--subset",
        choices=sorted(SUBSETS),
        default="medical_en",
        help="Predefined file bundle to download. Default: medical_en.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit SFT filenames under train/sft/ (overrides --subset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/apollo/qa"),
        help="Directory for raw JSON, per-file JSONL, and merged output.",
    )
    parser.add_argument(
        "--merged-name",
        default="merged_apollo_en_qa.jsonl",
        help="Filename for the merged JSONL output.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse already-downloaded raw JSON files in output-dir/raw.",
    )
    args = parser.parse_args()

    filenames = args.files if args.files else SUBSETS[args.subset]
    raw_dir = args.output_dir / "raw"
    jsonl_dir = args.output_dir / "jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    per_file_stats: list[dict[str, object]] = []
    jsonl_paths: list[Path] = []

    for filename in filenames:
        raw_target = raw_dir / filename
        jsonl_target = jsonl_dir / f"{Path(filename).stem}.jsonl"

        if args.skip_download and raw_target.exists():
            source_path = raw_target
            print(f"Reusing existing raw file: {source_path}")
        else:
            downloaded_path = download_file(filename, args.cache_dir)
            source_path = downloaded_path
            raw_target.write_bytes(downloaded_path.read_bytes())
            print(f"Saved raw copy to {raw_target}")

        row_count = json_array_to_jsonl(source_path, jsonl_target)
        jsonl_paths.append(jsonl_target)
        per_file_stats.append(
            {
                "filename": filename,
                "raw_path": str(raw_target),
                "jsonl_path": str(jsonl_target),
                "records": row_count,
            }
        )
        print(f"Converted {filename}: {row_count:,} records -> {jsonl_target}")

    merged_path = args.output_dir / args.merged_name
    merged_rows = merge_jsonl_files(jsonl_paths, merged_path)
    print(f"Merged {len(jsonl_paths)} files -> {merged_path} ({merged_rows:,} rows)")

    metadata = {
        "repo_id": REPO_ID,
        "subset": args.subset if not args.files else "custom",
        "files": filenames,
        "per_file": per_file_stats,
        "merged_path": str(merged_path),
        "merged_rows": merged_rows,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
