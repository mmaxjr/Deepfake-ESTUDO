"""Main desktop GUI window: 3 tabs (identity / scene / pipeline) + a shared
log console at the bottom. Run via `python gui.py` from the project root.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from .identity_tab import IdentityTab
from .log_console import LogConsole
from .pipeline_tab import PipelineTab
from .scene_tab import SceneTab

DATA_ROOT = "data"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ferramenta educacional de face-swap")
        self.geometry("900x760")
        self.minsize(760, 600)

        self.identity_name_var = tk.StringVar(value="")
        self.scene_path_var = tk.StringVar(value="")
        self.scene_dataset_var = tk.StringVar(value="custom")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        identity_tab = IdentityTab(notebook, data_root=DATA_ROOT, identity_name_var=self.identity_name_var)
        scene_tab = SceneTab(
            notebook,
            data_root=DATA_ROOT,
            scene_path_var=self.scene_path_var,
            scene_dataset_var=self.scene_dataset_var,
        )
        pipeline_tab = PipelineTab(
            notebook,
            data_root=DATA_ROOT,
            identity_name_var=self.identity_name_var,
            scene_path_var=self.scene_path_var,
            scene_dataset_var=self.scene_dataset_var,
        )

        notebook.add(identity_tab, text="1. Identidade")
        notebook.add(scene_tab, text="2. Cena")
        notebook.add(pipeline_tab, text="3. Pipeline")

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=False, padx=8, pady=(4, 8))
        self.log_console = LogConsole(log_frame)
        self.log_console.pack(fill="both", expand=True, padx=4, pady=4)

        logging.getLogger(__name__).info(
            "Pronto. Dica: preencha a aba 1 (identidade + consentimento) antes de treinar."
        )


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
