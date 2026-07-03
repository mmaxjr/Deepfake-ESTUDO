"""Thin entrypoint: `python infer.py --identity ... --scene ... --output ...`
Equivalent to `python cli.py infer ...`."""

from __future__ import annotations

import sys

from cli import app

if __name__ == "__main__":
    app(["infer"] + sys.argv[1:])
