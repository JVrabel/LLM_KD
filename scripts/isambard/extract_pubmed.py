#!/usr/bin/env python3
"""Extract PubMed abstracts from the NLM baseline XML into one JSONL per file.

Each output line is {"pmid", "text"}. Files that already have a sibling ``.ok``
marker are skipped, so the script resumes after a timeout.
"""

import argparse
import glob
import gzip
import json
import os
import xml.etree.ElementTree as ET


def local(tag):
    return tag.rsplit("}", 1)[-1]


def abstracts_in(path):
    with gzip.open(path, "rb") as handle:
        for _, article in ET.iterparse(handle, events=("end",)):
            if local(article.tag) != "PubmedArticle":
                continue
            pmid = None
            parts = []
            for element in article.iter():
                name = local(element.tag)
                if name == "PMID" and pmid is None and element.text:
                    pmid = element.text.strip()
                elif name == "AbstractText":
                    text = "".join(element.itertext()).strip()
                    if text:
                        label = element.attrib.get("Label")
                        parts.append(f"{label}: {text}" if label else text)
            article.clear()
            if pmid and parts:
                yield pmid, "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.xml_dir, "pubmed*.xml.gz")))
    print(f"{len(files)} baseline files", flush=True)
    total = 0
    for index, path in enumerate(files, start=1):
        name = os.path.basename(path).replace(".xml.gz", ".jsonl")
        dest = os.path.join(args.out_dir, name)
        if os.path.exists(dest + ".ok"):
            continue
        written = 0
        with open(dest, "w") as handle:
            for pmid, text in abstracts_in(path):
                handle.write(json.dumps({"pmid": pmid, "text": text}) + "\n")
                written += 1
        open(dest + ".ok", "w").close()
        total += written
        print(f"[{index}/{len(files)}] {name} abstracts {written}", flush=True)
    print(f"DONE wrote {total} abstracts this run", flush=True)


if __name__ == "__main__":
    main()
