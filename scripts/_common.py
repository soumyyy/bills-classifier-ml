"""Small helpers shared by the pipeline scripts.

Deliberately tiny and dependency-free, so importing it costs nothing and it
cannot become another place where training logic diverges. Scripts run as
`python scripts/<name>.py`, which puts this directory on sys.path, so
`from _common import require_file` resolves the same way
`import train as binary_train` already does in train_verifier_large.py.
"""

import hashlib
import subprocess
from pathlib import Path


def require_file(path: Path, hint: str) -> Path:
    """Return `path`, or exit with an actionable message if it is missing.

    Every load of a model or a split file used to assume the file was there.
    On a fresh clone - or after a step was skipped - that surfaced as a raw
    stack trace from deep inside Keras or csv, naming a path but not what
    produces it. `hint` is the command that creates the file.
    """
    if not path.exists():
        raise SystemExit(f"Missing {path}\n  Run: {hint}")
    return path


def file_sha256(path: Path) -> str:
    """Content hash of a file, or "missing" if it is not there."""
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    """Short commit the run was made from, or "unknown" outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def run_fingerprint(*, seed: int, manifest: Path, splits: Path, args: dict) -> dict:
    """Everything needed to tell whether two runs are the same run.

    train.py rewrites splits.csv in place on every run, so a model and the
    split file next to it can silently stop corresponding - which is exactly
    what happened: splits.csv held 920 test rows while the recorded metrics
    said 733, and nothing detected it. Recording the hashes lets the
    downstream scripts refuse to evaluate a model against a split it was
    never trained on.
    """
    return {
        "seed": seed,
        "git_revision": git_revision(),
        "manifest_sha256": file_sha256(manifest),
        "splits_sha256": file_sha256(splits),
        "args": args,
    }


def check_split_matches(recorded: dict | None, splits: Path, *, allow_mismatch: bool) -> None:
    """Abort if splits.csv is not the file the model was trained against.

    A mismatch means the metrics being produced describe a different split
    from the one the model saw - the numbers would look plausible and be
    meaningless. `allow_mismatch` is the deliberate override.
    """
    if not recorded:
        print(
            "  Note: this model records no split fingerprint, so it predates "
            "run tracking and cannot be checked against splits.csv."
        )
        return
    expected = recorded.get("splits_sha256")
    actual = file_sha256(splits)
    if expected is None or expected == actual:
        return
    message = (
        f"\n{splits} does not match the split this model was trained on.\n"
        f"  model recorded: {expected}\n"
        f"  on disk now:    {actual}\n"
        "  train.py rewrites splits.csv on every run, so re-training after this\n"
        "  model was saved will have replaced it. Any metric computed now would\n"
        "  describe a different split from the one the model saw."
    )
    if allow_mismatch:
        print(f"{message}\n  Continuing anyway (--allow-split-mismatch).")
        return
    raise SystemExit(f"{message}\n  Re-run training, or pass --allow-split-mismatch.")
