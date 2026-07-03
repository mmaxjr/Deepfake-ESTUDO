"""Thin entrypoint: `python train.py --identity ... --scene-cache ...`
Equivalent to `python cli.py train ...`."""

from __future__ import annotations

import sys

from cli import app

if __name__ == "__main__":
    app(["train"] + sys.argv[1:])
