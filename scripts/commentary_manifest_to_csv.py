#!/usr/bin/env python3
"""
Convert ``commentary_segments/audio_label_manifest.jsonl`` to a CSV
(id, frame, time, type, text, wav) for review or editing in a spreadsheet.

Optionally merge a notes file (``--notes``) with lines ``<id> <comment>`` into
``commentary_phrases_with_comments.csv`` (column ``user_comment``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def _parse_notes(path: Path) -> dict[int, str]:
    notes: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s+(.+)$", line)
        if m:
            notes[int(m.group(1))] = m.group(2).strip()
    return notes


def _write_phrases_with_comments(
    phrases_csv: Path,
    notes: dict[int, str],
    out: Path,
) -> int:
    rows: list[dict[str, str]] = []
    with phrases_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or []) + ["user_comment"]
        for row in r:
            i = int(row["id"])
            row["user_comment"] = notes.get(i, "")
            rows.append(row)
    seen = {int(x["id"]) for x in rows}
    for i in sorted(notes):
        if i not in seen:
            rows.append(
                {
                    "id": str(i),
                    "source_frame": "",
                    "time_sec": "",
                    "label": "",
                    "label_name": "",
                    "text": "",
                    "wav": "",
                    "user_comment": notes[i],
                }
            )
    rows.sort(key=lambda x: int(x["id"]))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        help="Path to audio_label_manifest.jsonl (default: <section>/commentary_segments/audio_label_manifest.jsonl)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV (default: next to manifest as commentary_phrases.csv)",
    )
    p.add_argument(
        "--section",
        type=Path,
        help="MARIO section directory containing commentary_segments/ (if manifest not given)",
    )
    p.add_argument(
        "--notes",
        type=Path,
        help="Optional text file: lines like '42 my note' (id + space + comment) merged into user_comment column",
    )
    p.add_argument(
        "--with-comments",
        type=Path,
        help="Output for phrases+notes (default: commentary_phrases_with_comments.csv next to main output)",
    )
    args = p.parse_args()

    if args.manifest is not None:
        manifest = args.manifest
    elif args.section is not None:
        manifest = Path(args.section) / "commentary_segments" / "audio_label_manifest.jsonl"
    else:
        p.error("pass manifest path, or --section /path/to/mario_1")
    if not manifest.is_file():
        print(f"not found: {manifest}", file=sys.stderr)
        return 1

    out = args.output
    if out is None:
        out = manifest.parent / "commentary_phrases.csv"

    rows: list[dict[str, str]] = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            sf = o.get("source_frame")
            rows.append(
                {
                    "id": str(o.get("id", "")),
                    "source_frame": "" if sf is None else str(int(sf)),
                    "time_sec": str(o.get("time_sec", "")),
                    "label": str(o.get("label", "")),
                    "label_name": str(o.get("label_name", "")),
                    "text": str(o.get("text", "")),
                    "wav": str(o.get("wav", "")),
                }
            )

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "source_frame", "time_sec", "label", "label_name", "text", "wav"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out}", flush=True)

    if args.notes is not None:
        if not args.notes.is_file():
            print(f"notes file not found: {args.notes}", file=sys.stderr)
            return 1
        wc = args.with_comments
        if wc is None:
            wc = out.parent / "commentary_phrases_with_comments.csv"
        n = _write_phrases_with_comments(out, _parse_notes(args.notes), wc)
        print(f"wrote {n} rows -> {wc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
