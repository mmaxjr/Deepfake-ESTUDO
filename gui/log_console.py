"""Thread-safe log console: a logging.Handler that queues records, polled by
the Tk event loop and rendered into a ScrolledText widget.

Long-running work (align/train/infer) runs on background threads so the
window doesn't freeze; `logging` calls from those threads land in the
queue and get drawn on the main thread via `after()`.
"""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import scrolledtext


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue[str]"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


class LogConsole(scrolledtext.ScrolledText):
    def __init__(self, master: tk.Misc, **kwargs):
        kwargs.setdefault("height", 12)
        kwargs.setdefault("state", "disabled")
        kwargs.setdefault("wrap", "word")
        super().__init__(master, **kwargs)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

        self.after(100, self._poll)

    def _poll(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.configure(state="normal")
            self.insert("end", line + "\n")
            self.see("end")
            self.configure(state="disabled")
        self.after(100, self._poll)

    def write_line(self, text: str) -> None:
        self.log_queue.put(text)
