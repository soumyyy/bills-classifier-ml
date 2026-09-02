"""Small helpers shared by the pipeline scripts.

Deliberately tiny and dependency-free, so importing it costs nothing and it
cannot become another place where training logic diverges. Scripts run as
`python scripts/<name>.py`, which puts this directory on sys.path, so
`from _common import require_file` resolves the same way
`import train as binary_train` already does in train_verifier_large.py.
"""

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
