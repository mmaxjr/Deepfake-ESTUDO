"""Desktop GUI entrypoint.

Run: python gui.py

The window itself only needs Tkinter (stdlib) + pyyaml. The "Cena" tab
needs numpy/opencv-python. The "Pipeline" tab (align/train/infer) needs the
full requirements.txt installed in a Python 3.10/3.11 environment -- see
README.md.
"""

from __future__ import annotations

from gui.app import main

if __name__ == "__main__":
    main()
