"""Respectful download of the public OULAD archive.

OULAD is published by The Open University's Knowledge Media Institute and is
also mirrored by the UCI Machine Learning Repository. This module fetches the
public archive, extracts the CSVs into ``data/raw``, and verifies them.

It is never required: if the files are already present, nothing is downloaded.
If the network is unavailable, the command prints the manual steps and exits
with a clear message rather than half-writing a broken dataset.

Usage
-----
    python -m src.data.download
    python -m src.data.download --force
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List

from src import config
from src.data import schema

_USER_AGENT = "Learner-Churn-Early-Warning-Dashboard/1.0 (local research prototype)"
MANUAL_STEPS = f"""
Manual download steps
---------------------
1. Open {config.OULAD_SOURCE_URL}
2. Download the dataset archive (also mirrored at:
   {config.OULAD_MIRROR_URL})
3. Extract these seven CSV files into: {config.RAW_DIR}
       {chr(10).join('       ' + f for f in config.OULAD_FILES)}
4. Re-run: python -m src.data.audit --data-dir data/raw
"""


def dataset_present(data_dir: Path) -> bool:
    return not schema.missing_files(data_dir)


def download_archive(url: str, dest: Path, timeout: int = 300) -> Path:
    """Stream the archive to ``dest`` with an explicit user agent."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)
    return dest


def extract_csvs(archive: Path, data_dir: Path) -> List[str]:
    """Extract only the OULAD CSVs, wherever they sit inside the archive."""
    wanted = {f.lower(): f for f in config.OULAD_FILES}
    extracted: List[str] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            base = Path(member).name.lower()
            if base in wanted:
                target = data_dir / wanted[base]
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(wanted[base])
    return sorted(extracted)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the OULAD dataset.")
    parser.add_argument("--data-dir", default=str(config.RAW_DIR))
    parser.add_argument("--url", default=config.OULAD_MIRROR_URL)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_present(data_dir) and not args.force:
        print(f"Dataset already present in {data_dir} - nothing to do.")
        return 0

    print(f"Downloading OULAD from {args.url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "oulad.zip"
        try:
            download_archive(args.url, archive)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"\nDownload failed: {exc}")
            print(MANUAL_STEPS)
            return 1

        size_mb = archive.stat().st_size / 1e6
        print(f"  downloaded {size_mb:.1f} MB")
        extracted = extract_csvs(archive, data_dir)

    missing = schema.missing_files(data_dir)
    print(f"  extracted {len(extracted)} files into {data_dir}")
    if missing:
        print(f"  WARNING: still missing {missing}")
        print(MANUAL_STEPS)
        return 1

    print("  verifying dataset ...")
    schema.check_dataset(data_dir)
    print("  dataset verified.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
