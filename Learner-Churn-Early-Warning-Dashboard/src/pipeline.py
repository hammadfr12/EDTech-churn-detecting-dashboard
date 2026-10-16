"""One-command pipeline: audit -> features -> train -> report.

This is the documented run path. Everything the dashboard needs is produced
by this command, in order, with each stage failing loudly rather than
half-writing artifacts.

Usage
-----
    python -m src.pipeline                 # use data/raw, cutoff week 4
    python -m src.pipeline --download      # fetch OULAD first if missing
    python -m src.pipeline --skip-train    # features only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from src import config
from src.data import audit as data_audit
from src.data import schema
from src.features import build as feature_build
from src.model import train as model_train
from src.analytics import report


def _stage(name: str) -> float:
    print(f"\n{'─' * 62}\n▶ {name}\n{'─' * 62}")
    return time.time()


def _done(started: float) -> None:
    print(f"  ✓ completed in {time.time() - started:.1f}s")


def run(
    data_dir: Path,
    cutoff_week: int = config.DEFAULT_CUTOFF_WEEK,
    target_key: str = config.DEFAULT_TARGET,
    skip_train: bool = False,
    download: bool = False,
) -> int:
    config.ensure_dirs()

    # --- dataset acquisition ------------------------------------------------
    if download:
        started = _stage("Downloading OULAD")
        from src.data import download as data_download

        if data_download.main(["--data-dir", str(data_dir)]) != 0:
            print("\nDownload failed. See the manual steps above, then re-run.")
            return 1
        _done(started)

    # --- 1. validate --------------------------------------------------------
    started = _stage("Validating the raw dataset")
    try:
        schema.check_dataset(data_dir)
    except schema.DatasetError as exc:
        print(f"\n✗ {exc}")
        print("\nTo fetch the data automatically:")
        print("    python -m src.pipeline --download")
        return 1
    print("  ✓ dataset complete and structurally valid")
    _done(started)

    # --- 2. audit -----------------------------------------------------------
    started = _stage("Auditing the data")
    audit = data_audit.run_audit(data_dir)
    data_audit.print_summary(audit)
    data_audit.write_audit(audit)
    _done(started)

    # --- 3. features --------------------------------------------------------
    started = _stage("Building SQL feature models")
    weeks = sorted({cutoff_week, config.COMPARISON_CUTOFF_WEEK, config.DEFAULT_CUTOFF_WEEK})
    feature_build.build(data_dir, weeks)
    _done(started)

    # --- 4. model -----------------------------------------------------------
    if skip_train:
        print("\n▶ Skipping training (--skip-train)")
    else:
        started = _stage(f"Training and evaluating the baseline model (week {cutoff_week})")
        metrics = model_train.train(cutoff_week=cutoff_week, target_key=target_key)
        model_train.print_report(metrics)
        _done(started)

    # --- 5. report ----------------------------------------------------------
    started = _stage("Writing the build report")
    report.write_build_report()
    _done(started)

    print(f"\n{'═' * 62}")
    print("Pipeline complete. Start the dashboard with:")
    print("    streamlit run app/Home.py")
    print(f"{'═' * 62}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full Learner Churn pipeline: audit, features, model, report."
    )
    parser.add_argument("--data-dir", default=str(config.RAW_DIR))
    parser.add_argument("--cutoff-week", type=int, default=config.DEFAULT_CUTOFF_WEEK)
    parser.add_argument("--target", default=config.DEFAULT_TARGET, choices=list(config.TARGETS))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the OULAD archive first if the raw files are missing.",
    )
    args = parser.parse_args(argv)

    return run(
        data_dir=Path(args.data_dir),
        cutoff_week=args.cutoff_week,
        target_key=args.target,
        skip_train=args.skip_train,
        download=args.download,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
