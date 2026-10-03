"""Verify an exported residue session through Lean kernel reduction."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hyperreals import ReplayVerificationError, verify_export

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="directory containing snapshot.json and Replay.lean",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="kernel replay timeout in seconds"
    )
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        verified = verify_export(
            args.directory, project_root=ROOT, timeout=args.timeout
        )
    except (ReplayVerificationError, ValueError, OSError) as error:
        parser.exit(1, f"Replay verification failed: {error}\n")
    print(f"Kernel replay verified in {time.perf_counter() - start:.3f}s.")
    print(f"Snapshot SHA-256: {verified.snapshot_sha256}")
    print(f"Lean source SHA-256: {verified.source_sha256}")
    print(f"Axiom dependencies: {', '.join(sorted(verified.axioms)) or 'none'}")


if __name__ == "__main__":
    main()
