"""Build the bundled journal-quartile lookup from SCImago export CSVs.

SCImago exports are semicolon-delimited and use either a "SJR Best Quartile"
column (Subject Area exports) or a "SJR Quartile" column (Subject Category
exports). A journal can appear in several category files with different
quartiles; we keep the BEST (Q1 > Q2 > Q3 > Q4) seen across all inputs, which
is the fairest single proxy for journal standing.

Usage:
    python scripts/build_quartile_table.py "/path/to/SCImago CSV dir" \
        [data/journal_quartiles.csv]

Re-run yearly when SCImago publishes new rankings. The output is a plain
comma-delimited CSV (journal,quartile,quartile_source) that the app loads via
parse_quartile_overrides().
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

_RANK = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def normalize_journal(journal: str) -> str:
    # Must match paper_finder.normalize_journal exactly.
    return re.sub(r"[^a-z0-9]+", " ", journal.lower()).strip()


def normalize_quartile(value: str) -> str:
    match = re.search(r"q\s*([1-4])", (value or "").strip().lower())
    return f"Q{match.group(1)}" if match else ""


def build(source_dir: Path, output_path: Path) -> None:
    # normalized_journal -> (rank, original_title, quartile)
    best: dict[str, tuple[int, str, str]] = {}
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {source_dir}")

    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for row in reader:
                title = (row.get("Title") or "").strip()
                quartile = normalize_quartile(
                    row.get("SJR Best Quartile") or row.get("SJR Quartile") or ""
                )
                if not title or not quartile:
                    continue
                key = normalize_journal(title)
                if not key:
                    continue
                rank = _RANK[quartile]
                current = best.get(key)
                if current is None or rank < current[0]:
                    best[key] = (rank, title, quartile)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["journal", "quartile", "quartile_source"])
        for _key, (_rank, title, quartile) in sorted(best.items()):
            writer.writerow([title, quartile, "SCImago 2025"])

    print(f"Wrote {len(best)} journals to {output_path} from {len(files)} source file(s).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: build_quartile_table.py <source_dir> [output_csv]")
    src = Path(sys.argv[1]).expanduser()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/journal_quartiles.csv")
    build(src, out)
