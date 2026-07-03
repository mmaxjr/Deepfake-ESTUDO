"""Tab 1: manage a local identity folder (your own consented photos/video)
and its consent.yaml manifest.

Requires: pyyaml, numpy, opencv-python (for datasets.identity_dataset).
Does NOT require torch/insightface.
"""

from __future__ import annotations

import logging
import shutil
import tkinter as tk
from datetime import date
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

logger = logging.getLogger(__name__)

IMAGE_FILETYPES = [("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos os arquivos", "*.*")]
VIDEO_FILETYPES = [("Vídeos", "*.mp4 *.mov *.avi *.mkv *.webm"), ("Todos os arquivos", "*.*")]


class IdentityTab(ttk.Frame):
    def __init__(self, master, data_root: str, identity_name_var: tk.StringVar):
        super().__init__(master)
        self.data_root = Path(data_root)
        self.identity_name_var = identity_name_var

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(top, text="Nome da identidade (pasta):").pack(side="left")
        ttk.Entry(top, textvariable=self.identity_name_var, width=30).pack(side="left", padx=6)
        ttk.Button(top, text="Carregar / Criar pasta", command=self._load_identity).pack(side="left", padx=6)

        files_frame = ttk.LabelFrame(self, text="Fotos / vídeos da identidade")
        files_frame.pack(fill="both", expand=True, padx=10, pady=6)

        self.file_listbox = tk.Listbox(files_frame, height=8)
        self.file_listbox.pack(fill="both", expand=True, side="left", padx=(6, 0), pady=6)
        scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=self.file_listbox.yview)
        scrollbar.pack(side="left", fill="y", pady=6)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)

        btns = ttk.Frame(files_frame)
        btns.pack(side="left", fill="y", padx=8, pady=6)
        ttk.Button(btns, text="+ Adicionar fotos...", command=self._add_photos).pack(fill="x", pady=2)
        ttk.Button(btns, text="+ Adicionar vídeo...", command=self._add_video).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remover selecionado(s)", command=self._remove_selected).pack(fill="x", pady=2)

        consent_frame = ttk.LabelFrame(self, text="Consentimento (obrigatório antes de treinar)")
        consent_frame.pack(fill="x", padx=10, pady=6)

        row1 = ttk.Frame(consent_frame)
        row1.pack(fill="x", padx=6, pady=4)
        ttk.Label(row1, text="Nome do titular:", width=16).pack(side="left")
        self.subject_name_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.subject_name_var, width=40).pack(side="left", padx=4)

        ttk.Label(row1, text="Data:", width=6).pack(side="left", padx=(12, 0))
        self.date_var = tk.StringVar(value=date.today().isoformat())
        ttk.Entry(row1, textvariable=self.date_var, width=12).pack(side="left", padx=4)

        ttk.Label(consent_frame, text="Declaração de consentimento:").pack(anchor="w", padx=6)
        self.statement_text = tk.Text(consent_frame, height=3, wrap="word")
        self.statement_text.pack(fill="x", padx=6, pady=(0, 4))
        self.statement_text.insert(
            "1.0",
            "Eu sou a pessoa nas fotos/vídeos desta pasta e consinto que minha "
            "imagem seja usada para treinar esta ferramenta de face-swap para "
            "um projeto pessoal, não-comercial.",
        )

        self.consent_confirmed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            consent_frame,
            text="Confirmo que as informações acima são verdadeiras (consent_confirmed)",
            variable=self.consent_confirmed_var,
        ).pack(anchor="w", padx=6, pady=(0, 4))

        actions = ttk.Frame(consent_frame)
        actions.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(actions, text="Salvar consentimento", command=self._save_consent).pack(side="left")
        self.status_label = ttk.Label(actions, text="Status: não verificado")
        self.status_label.pack(side="left", padx=12)

    def _identity_folder(self) -> Path | None:
        name = self.identity_name_var.get().strip()
        if not name:
            messagebox.showwarning("Identidade", "Informe um nome de identidade primeiro.")
            return None
        return self.data_root / "identities" / name

    def _load_identity(self) -> None:
        folder = self._identity_folder()
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        self._refresh_file_list(folder)
        self._load_existing_consent(folder)
        logger.info("Identidade carregada: %s", folder)

    def _refresh_file_list(self, folder: Path) -> None:
        self.file_listbox.delete(0, "end")
        try:
            from datasets.common import list_images, list_videos

            media = sorted(list_images(folder) + list_videos(folder))
        except ImportError:
            media = sorted(
                p for p in folder.iterdir() if p.is_file() and p.suffix.lower() != ".yaml"
            )
        for path in media:
            self.file_listbox.insert("end", path.name)

    def _load_existing_consent(self, folder: Path) -> None:
        manifest_path = folder / "consent.yaml"
        if not manifest_path.exists():
            return
        import yaml

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.subject_name_var.set(raw.get("subject_name", ""))
        self.date_var.set(raw.get("date", date.today().isoformat()))
        self.statement_text.delete("1.0", "end")
        self.statement_text.insert("1.0", raw.get("statement", ""))
        self.consent_confirmed_var.set(bool(raw.get("consent_confirmed", False)))
        self._refresh_status(folder)

    def _add_photos(self) -> None:
        folder = self._identity_folder()
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        paths = filedialog.askopenfilenames(title="Selecionar fotos", filetypes=IMAGE_FILETYPES)
        for src in paths:
            shutil.copy2(src, folder / Path(src).name)
        if paths:
            logger.info("Adicionadas %d foto(s) em %s", len(paths), folder)
        self._refresh_file_list(folder)

    def _add_video(self) -> None:
        folder = self._identity_folder()
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        src = filedialog.askopenfilename(title="Selecionar vídeo", filetypes=VIDEO_FILETYPES)
        if src:
            shutil.copy2(src, folder / Path(src).name)
            logger.info("Adicionado vídeo %s em %s", Path(src).name, folder)
        self._refresh_file_list(folder)

    def _remove_selected(self) -> None:
        folder = self._identity_folder()
        if folder is None:
            return
        selection = self.file_listbox.curselection()
        if not selection:
            return
        names = [self.file_listbox.get(i) for i in selection]
        if not messagebox.askyesno("Remover", f"Remover {len(names)} arquivo(s) de {folder}?"):
            return
        for name in names:
            path = folder / name
            if path.exists():
                path.unlink()
        logger.info("Removido(s) %d arquivo(s) de %s", len(names), folder)
        self._refresh_file_list(folder)

    def _save_consent(self) -> None:
        folder = self._identity_folder()
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)

        import yaml

        data = {
            "consent_confirmed": bool(self.consent_confirmed_var.get()),
            "subject_name": self.subject_name_var.get().strip(),
            "date": self.date_var.get().strip(),
            "statement": self.statement_text.get("1.0", "end").strip(),
        }
        with open(folder / "consent.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)

        logger.info("consent.yaml salvo em %s", folder)
        self._refresh_status(folder)

    def _refresh_status(self, folder: Path) -> None:
        try:
            from datasets.identity_dataset import ConsentNotConfirmedError, load_consent_manifest
        except ImportError:
            self.status_label.configure(text="Status: não verificado (instale pyyaml/numpy/opencv-python)")
            return

        try:
            manifest = load_consent_manifest(folder)
            self.status_label.configure(text=f"Status: válido (titular: {manifest.subject_name})")
        except ConsentNotConfirmedError as exc:
            self.status_label.configure(text=f"Status: pendente — {exc}")
