"""Tab 2: choose the scene/driving video source -- a custom video/folder, or
a locally-downloaded research dataset (FaceForensics++ / Celeb-DF / DFDC).

This tab never downloads or scrapes any dataset -- it only points at a local
path and (optionally) shows the official registration instructions.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

logger = logging.getLogger(__name__)

DATASET_OPTIONS = ["custom", "faceforensics", "celebdf", "dfdc"]


class SceneTab(ttk.Frame):
    def __init__(self, master, data_root: str, scene_path_var: tk.StringVar, scene_dataset_var: tk.StringVar):
        super().__init__(master)
        self.data_root = data_root
        self.scene_path_var = scene_path_var
        self.scene_dataset_var = scene_dataset_var
        if not self.scene_dataset_var.get():
            self.scene_dataset_var.set("custom")

        row = ttk.Frame(self)
        row.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(row, text="Fonte da cena:").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.scene_dataset_var, values=DATASET_OPTIONS, state="readonly", width=16)
        combo.pack(side="left", padx=6)
        ttk.Button(row, text="Ver instruções de acesso", command=self._show_dataset_info).pack(side="left", padx=6)

        path_row = ttk.Frame(self)
        path_row.pack(fill="x", padx=10, pady=4)
        ttk.Label(path_row, text="Caminho (vídeo, pasta, ou raiz do dataset):").pack(anchor="w")
        entry_row = ttk.Frame(path_row)
        entry_row.pack(fill="x", pady=4)
        ttk.Entry(entry_row, textvariable=self.scene_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(entry_row, text="Arquivo...", command=self._browse_file).pack(side="left", padx=4)
        ttk.Button(entry_row, text="Pasta...", command=self._browse_folder).pack(side="left")

        ttk.Button(self, text="Testar / contar vídeos", command=self._count_videos).pack(anchor="w", padx=10, pady=6)

        help_text = (
            "custom: um arquivo de vídeo único, ou uma pasta com vários vídeos seus.\n"
            "faceforensics / celebdf / dfdc: aponte para a pasta raiz já baixada por você "
            "(esta ferramenta não baixa nem faz scraping desses datasets)."
        )
        ttk.Label(self, text=help_text, foreground="#666666", justify="left").pack(anchor="w", padx=10, pady=(0, 10))

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecionar vídeo de cena",
            filetypes=[("Vídeos", "*.mp4 *.mov *.avi *.mkv *.webm"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.scene_path_var.set(path)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Selecionar pasta de cena / raiz do dataset")
        if path:
            self.scene_path_var.set(path)

    def _show_dataset_info(self) -> None:
        name = self.scene_dataset_var.get()
        if name == "custom":
            messagebox.showinfo("custom", "Vídeo/pasta próprios: sem necessidade de registro.")
            return
        try:
            from datasets.dataset_info import get_dataset_info

            info = get_dataset_info(name)
            text = (
                f"{info['display_name']}\n\n"
                f"Acesso: {info['how_to_get_access']}\n\n"
                f"Layout esperado:\n{info['expected_layout']}"
            )
            messagebox.showinfo(info["display_name"], text)
        except Exception as exc:  # noqa: BLE001 - surface any error to the user
            messagebox.showerror("Erro", str(exc))

    def _count_videos(self) -> None:
        name = self.scene_dataset_var.get()
        root = self.scene_path_var.get().strip()
        if not root:
            messagebox.showwarning("Cena", "Informe um caminho primeiro.")
            return
        try:
            from datasets.scene_dataset import build_scene_dataset

            scene = build_scene_dataset(name, root)
            count = scene.count()
            logger.info("Dataset '%s' em %s: %d vídeo(s) encontrado(s)", name, root, count)
            messagebox.showinfo("Resultado", f"{count} vídeo(s) encontrado(s) em '{root}'.")
        except ImportError:
            messagebox.showerror(
                "Dependência ausente",
                "Instale numpy/opencv-python (pip install -r requirements.txt) para usar esta função.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha ao indexar cena")
            messagebox.showerror("Erro", str(exc))
