"""Download public Parquet files, sample whole source groups, keep raw rows.

No hosted inference, no dataset remote Python script, and no paid API.
Default is a dry plan. Pass --download to fetch, or --local-file to reuse bytes.
The outputs are raw staging data, not the experiment public/private schema.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


REGISTRY = {
    "e2e": {
        "repo_id": "GEM/e2e_nlg",
        "revision": "refs/convert/parquet",
        "file_pattern": "default/{split}/0000.parquet",
        "splits": ["train", "validation", "test"],
        "license": "cc-by-sa-4.0",
        "required_columns": ["meaning_representation", "target", "references"],
    },
    "webnlg": {
        "repo_id": "GEM/web_nlg",
        "revision": "refs/convert/parquet",
        "file_pattern": "en/{split}/0000.parquet",
        "splits": ["train", "validation", "test"],
        "license": "cc-by-nc-4.0",
        "required_columns": ["input", "target", "references"],
    },
    "squad2": {
        "repo_id": "rajpurkar/squad_v2",
        "revision": "main",
        "file_pattern": "squad_v2/{split}-00000-of-00001.parquet",
        "splits": ["train", "validation"],
        "license": "cc-by-sa-4.0",
        "required_columns": ["context", "question", "answers", "id"],
    },
}


def plain(value):
    """Convert numpy-backed Parquet cells to ordinary JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported Parquet cell type: {type(value)!r}")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def digest(value):
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def group_key(dataset, row):
    if dataset == "e2e":
        raw = str(row["meaning_representation"])
        matches = list(re.finditer(r"([^,\[\]]+)\[([^\]]*)\]", raw))
        if not matches:
            raise ValueError(f"Cannot parse E2E MR: {raw!r}")
        cursor = 0
        for match in matches:
            if raw[cursor:match.start()].strip(" ,\t\r\n"):
                raise ValueError(f"Unparsed E2E MR text: {raw!r}")
            cursor = match.end()
        if raw[cursor:].strip(" ,\t\r\n"):
            raise ValueError(f"Unparsed E2E MR suffix: {raw!r}")
        pairs = sorted((normalize(m[1]).casefold(), normalize(m[2])) for m in matches)
        return "e2e_" + digest(pairs)
    if dataset == "webnlg":
        triples = row["input"]
        if not isinstance(triples, list) or not triples:
            raise ValueError("WebNLG input must be a nonempty list")
        canonical = []
        for triple in triples:
            parts = [normalize(part) for part in str(triple).split("|")]
            if len(parts) != 3:
                raise ValueError(f"Not a subject|predicate|object triple: {triple!r}")
            canonical.append(parts)
        return "webnlg_" + digest(sorted(canonical))
    if dataset == "squad2":
        context = normalize(row["context"])
        if not context:
            raise ValueError("Empty SQuAD context")
        return "squad2_" + digest(context)
    raise ValueError(dataset)


def candidate_text(dataset, row):
    if dataset == "squad2":
        return str(row["context"])
    target = row.get("target")
    if isinstance(target, str) and target.strip():
        return target
    refs = row.get("references") or []
    if refs and isinstance(refs[0], str) and refs[0].strip():
        return refs[0]
    raise ValueError("No nonempty target or first reference in this row")


def group_rows(dataset, rows, max_groups, seed, min_words=0, max_words=None):
    """Select groups before seeing editing outcomes; retain every row in each group."""
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[group_key(dataset, row)].append(index)
    candidates = []
    for gid, indices in sorted(buckets.items()):
        valid_sources = []
        for index in indices:
            try:
                text = candidate_text(dataset, rows[index])
            except ValueError:
                continue
            words = len(text.split())
            if words >= min_words and (max_words is None or words <= max_words):
                valid_sources.append(index)
        if valid_sources:
            # Stable and independent of model/evaluator scores.
            source_index = valid_sources[0]
            candidates.append((gid, indices, source_index))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = sorted(candidates[:max_groups], key=lambda item: item[0])
    groups, selected_rows = [], []
    for gid, indices, source_index in chosen:
        text = candidate_text(dataset, rows[source_index])
        groups.append({
            "dataset": dataset,
            "group_id": gid,
            "source_row_index": source_index,
            "row_indices": indices,
            "row_count": len(indices),
            "candidate_source_text": text,
            "candidate_source_words": len(text.split()),
            "note": "Raw staging group; validate source facts before experiment normalization.",
        })
        for index in indices:
            selected_rows.append({
                "group_id": gid, "original_row_index": index,
                "raw": rows[index],
            })
    return groups, selected_rows, len(buckets), len(candidates)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolved_revision(path):
    parts = Path(path).parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-groups", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--revision", help="Pinned commit SHA; defaults to registry revision")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local-file", type=Path, help="Existing Parquet; no network request")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    cfg = REGISTRY[args.dataset]
    if args.split not in cfg["splits"]:
        parser.error(f"{args.dataset} has available splits {cfg['splits']}")
    if args.max_groups < 1 or args.min_words < 0:
        parser.error("max-groups must be positive; min-words must be nonnegative")
    if args.max_words is not None and args.max_words < args.min_words:
        parser.error("max-words must be >= min-words")
    if args.download and args.local_file:
        parser.error("Choose --download or --local-file, not both")
    revision = args.revision or cfg["revision"]
    filename = cfg["file_pattern"].format(split=args.split)
    plan = {
        "dataset": args.dataset, "repo_id": cfg["repo_id"], "split": args.split,
        "requested_revision": revision, "filename": filename,
        "license": cfg["license"], "max_groups": args.max_groups, "seed": args.seed,
        "min_words": args.min_words, "max_words": args.max_words,
        "hosted_inference": False,
    }
    if not args.download and not args.local_file:
        print(json.dumps({"mode": "plan_only", **plan}, indent=2))
        return
    if args.local_file:
        path = args.local_file.resolve()
        if not path.is_file():
            parser.error(f"Local file not found: {path}")
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(
            repo_id=cfg["repo_id"], repo_type="dataset", filename=filename,
            revision=revision, token=False,
        ))
    import pandas as pd
    frame = pd.read_parquet(path)
    missing = sorted(set(cfg["required_columns"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset schema mismatch, missing columns: {missing}")
    rows = [plain(row) for row in frame.to_dict(orient="records")]
    groups, selected, all_groups, eligible = group_rows(
        args.dataset, rows, args.max_groups, args.seed, args.min_words, args.max_words
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.split}_seed{args.seed}"
    group_path = args.out / f"{stem}_groups.jsonl"
    rows_path = args.out / f"{stem}_raw_rows.jsonl"
    manifest_path = args.out / f"{stem}_manifest.json"
    write_jsonl(group_path, groups)
    write_jsonl(rows_path, selected)
    manifest = {
        **plan, "resolved_revision": resolved_revision(path),
        "download_sha256": file_hash(path), "raw_columns": list(frame.columns),
        "total_rows": len(rows), "total_groups": all_groups,
        "eligible_groups": eligible, "selected_groups": len(groups),
        "selected_rows": len(selected),
        "groups_sha256": file_hash(group_path), "rows_sha256": file_hash(rows_path),
        "group_ids": [group["group_id"] for group in groups],
        "experiment_split_assigned": False,
        "source_fact_validation_completed": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": "saved_raw_staging", "groups": str(group_path),
        "raw_rows": str(rows_path), "manifest": str(manifest_path),
        "selected_groups": len(groups), "selected_rows": len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
