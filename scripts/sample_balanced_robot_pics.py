#!/usr/bin/env python3
"""
Reduce a large robot_pics folder to a balanced subset for CNN training.

Uses labels.json to stratify by class. Within each class, filenames like
``12345-7.png`` (frame_id-track_id) are sorted and sampled **evenly in time**
so consecutive near-duplicate frames are dropped.

If there are **no** labeled files on disk but many unlabeled crops, ``-n`` keeps
that many images with **time spacing** (no class balance possible until you label).

With labels, pass ``--keep-unlabeled-random K`` to also keep K extra unlabeled files.

Examples:
    dry-run plan:
        python scripts/sample_balanced_robot_pics.py -i training/robot_pics -n 1000 --dry-run

    move the rest to training/robot_pics_discarded/:
        python scripts/sample_balanced_robot_pics.py -i training/robot_pics -n 1000 --move-discarded

    delete unselected (irreversible):
        python scripts/sample_balanced_robot_pics.py -i training/robot_pics -n 1000 --delete-rest
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


_FRAME_RE = re.compile(r"^(\d+)-(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def _parse_frame_sort_key(name: str) -> Tuple[int, int]:
    m = _FRAME_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def list_images(d: Path) -> List[Path]:
    out = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        out.extend(d.glob(pat))
    return sorted(out, key=lambda p: p.name.lower())


def load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def allocate_quotas(class_counts: Dict[str, int], n_total: int) -> Dict[str, int]:
    """Fair per-class targets that never exceed available counts."""
    classes = sorted(c for c, n in class_counts.items() if n > 0)
    k = len(classes)
    if k == 0:
        return {}
    quotas = {c: 0 for c in classes}
    # First pass: ideal balanced caps
    for i, c in enumerate(classes):
        want = n_total // k + (1 if i < (n_total % k) else 0)
        quotas[c] = min(class_counts[c], want)
    short = n_total - sum(quotas.values())
    # Fill deficit round-robin among classes with spare capacity
    p = 0
    guard = 0
    while short > 0 and guard < n_total * k + 10:
        c = classes[p % k]
        p += 1
        guard += 1
        if quotas[c] < class_counts[c]:
            quotas[c] += 1
            short -= 1
    return quotas


def pick_spaced(paths: List[Path], quota: int, rng: random.Random) -> List[Path]:
    """Evenly spaced picks along sorted-by-frame list (temporal diversity)."""
    if quota <= 0 or not paths:
        return []
    n = len(paths)
    if n <= quota:
        return paths.copy()
    if quota == 1:
        return [paths[n // 2]]
    idxs = [int(round(i * (n - 1) / (quota - 1))) for i in range(quota)]
    seen_i: set = set()
    out: List[Path] = []
    for i in idxs:
        i = min(max(0, i), n - 1)
        if i not in seen_i:
            seen_i.add(i)
            out.append(paths[i])
    while len(out) < quota:
        j = rng.randrange(n)
        if j not in seen_i:
            seen_i.add(j)
            out.append(paths[j])
    return out[:quota]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, required=True, help="robot_pics directory")
    ap.add_argument("-n", "--keep", type=int, required=True, help="Total images to keep")
    ap.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="labels.json path (default: INPUT/labels.json)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for tie-breaks")
    ap.add_argument(
        "--keep-unlabeled-random",
        type=int,
        default=0,
        metavar="K",
        help="Also keep K unlabeled files (time-spaced random), added on top of -n",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--move-discarded",
        action="store_true",
        help="Move unselected files to INPUT/../<name>_discarded/",
    )
    g.add_argument(
        "--delete-rest",
        action="store_true",
        help="Permanently delete unselected files",
    )
    args = ap.parse_args()
    inp = args.input.resolve()
    if not inp.is_dir():
        print(f"Not a directory: {inp}", file=sys.stderr)
        return 1

    labels_path = args.labels or (inp / "labels.json")
    labels = load_labels(labels_path)
    rng = random.Random(args.seed)

    all_images = list_images(inp)
    if not all_images:
        print(f"No images in {inp}")
        return 0

    by_name = {p.name: p for p in all_images}
    # Labeled files that still exist on disk
    by_class: Dict[str, List[Path]] = {}
    for fname, cls in labels.items():
        p = by_name.get(fname)
        if p is not None:
            by_class.setdefault(cls, []).append(p)

    for cls in by_class:
        by_class[cls].sort(key=lambda p: _parse_frame_sort_key(p.name))

    class_counts = {c: len(v) for c, v in by_class.items()}
    labeled_paths = {p for paths in by_class.values() for p in paths}
    unlabeled = [p for p in all_images if p not in labeled_paths]

    available_labeled = sum(class_counts.values())
    n_keep = args.keep
    extra: List[Path] = []

    if available_labeled == 0:
        if not unlabeled:
            print(
                "No labeled files and no image files. Nothing to do.",
                file=sys.stderr,
            )
            return 1
        # Time-spaced sample from unlabeled only (cannot balance by class yet).
        unlabeled.sort(key=lambda p: _parse_frame_sort_key(p.name))
        take_n = min(n_keep, len(unlabeled))
        selected = pick_spaced(unlabeled, take_n, rng)
        quotas = {}
        if args.keep_unlabeled_random > 0:
            print(
                "Note: with no labels, -n already subsamples unlabeled; "
                "--keep-unlabeled-random is ignored.",
                file=sys.stderr,
            )
    else:
        n_effective = min(n_keep, available_labeled)
        if n_effective < n_keep:
            print(
                f"Note: only {available_labeled} labeled files on disk; "
                f"keeping {n_effective} (requested {n_keep}).",
                file=sys.stderr,
            )
        quotas = allocate_quotas(class_counts, n_effective)
        selected = []
        for cls, q in sorted(quotas.items()):
            pool = by_class.get(cls, [])
            take = pick_spaced(pool, q, rng)
            selected.extend(take)

        # Extra unlabeled slots on top of balanced labeled picks
        if args.keep_unlabeled_random > 0 and unlabeled:
            pool_u = [p for p in unlabeled if p not in selected]
            pool_u.sort(key=lambda p: _parse_frame_sort_key(p.name))
            extra = pick_spaced(pool_u, args.keep_unlabeled_random, rng)

    keep_set = set(selected) | set(extra)
    discard = [p for p in all_images if p not in keep_set]

    print(f"Input dir:     {inp}")
    print(f"Images found:  {len(all_images)}")
    print(f"Labeled:       {len(labeled_paths)}  Unlabeled: {len(unlabeled)}")
    if available_labeled > 0:
        print(
            f"Keep (target): {n_keep} labeled stratified → "
            f"kept {len(set(selected))} labeled"
        )
    else:
        print(f"Keep: {len(selected)} unlabeled (time-spaced), target was {n_keep}")
    if extra:
        print(f"Extra unlabeled: {len(extra)}")
    print(f"Total keep:    {len(keep_set)}")
    print(f"Discard:       {len(discard)}")
    for cls in sorted(quotas.keys()):
        got = sum(1 for p in selected if labels.get(p.name) == cls)
        print(f"  {cls}: quota {quotas[cls]} → picked {got} (available {class_counts.get(cls, 0)})")

    if args.dry_run:
        print("\n[--dry-run] No files changed.")
        return 0

    if not discard:
        print("Nothing to discard.")
        return 0

    if not args.move_discarded and not args.delete_rest:
        print("\nChoose --move-discarded or --delete-rest (or --dry-run).", file=sys.stderr)
        return 1

    if args.move_discarded:
        discarded_dir = inp.parent / f"{inp.name}_discarded"
        discarded_dir.mkdir(parents=True, exist_ok=True)
        for p in discard:
            dest = discarded_dir / p.name
            if dest.exists():
                stem, suf = p.stem, p.suffix
                dest = discarded_dir / f"{stem}_dup{rng.randint(0, 99999)}{suf}"
            shutil.move(str(p), str(dest))
        print(f"Moved {len(discard)} files → {discarded_dir}")
    else:
        for p in discard:
            p.unlink(missing_ok=True)
        print(f"Deleted {len(discard)} files.")

    # Optional: prune labels.json to only kept labeled files
    if labels and labels_path.exists():
        new_labels = {k: v for k, v in labels.items() if k in by_name and by_name[k] in keep_set}
        if len(new_labels) != len(labels):
            bak = labels_path.with_suffix(".json.bak")
            shutil.copy2(labels_path, bak)
            with open(labels_path, "w") as f:
                json.dump(new_labels, f, indent=2)
            print(f"Updated {labels_path} ({len(new_labels)} entries); backup: {bak}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
