"""Tab 3: run the actual pipeline -- align, train, infer.

Each action runs on a background thread so the window stays responsive;
progress/errors are surfaced through the shared logging-based console (see
log_console.py) rather than blocking dialogs. Only one action runs at a
time (buttons are disabled while busy).
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

logger = logging.getLogger(__name__)


class PipelineTab(ttk.Frame):
    def __init__(
        self,
        master,
        data_root: str,
        identity_name_var: tk.StringVar,
        scene_path_var: tk.StringVar,
        scene_dataset_var: tk.StringVar,
    ):
        super().__init__(master)
        self.data_root = Path(data_root)
        self.identity_name_var = identity_name_var
        self.scene_path_var = scene_path_var
        self.scene_dataset_var = scene_dataset_var
        self._busy = False
        self._buttons: list[ttk.Button] = []

        self.resolution_var = tk.StringVar(value="128")
        self.epochs_var = tk.StringVar(value="200")
        self.batch_size_var = tk.StringVar(value="16")
        self.lr_var = tk.StringVar(value="0.0002")
        self.checkpoint_path_var = tk.StringVar(value="")
        self.infer_scene_var = tk.StringVar(value="")
        self.infer_output_var = tk.StringVar(value="")
        self.blend_var = tk.StringVar(value="feather")
        self.keep_audio_var = tk.BooleanVar(value=True)

        self._build_align_section()
        self._build_train_section()
        self._build_infer_section()

    # -- section builders ---------------------------------------------------

    def _build_align_section(self) -> None:
        frame = ttk.LabelFrame(self, text="1. Alinhamento (detecção facial + cache de crops)")
        frame.pack(fill="x", padx=10, pady=6)

        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Resolução:").pack(side="left")
        ttk.Combobox(row, textvariable=self.resolution_var, values=["128", "256"], state="readonly", width=6).pack(
            side="left", padx=6
        )

        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=6, pady=(0, 6))
        self._add_button(btns, "Alinhar identidade", self._on_align_identity)
        self._add_button(btns, "Alinhar cena", self._on_align_scene)

    def _build_train_section(self) -> None:
        frame = ttk.LabelFrame(self, text="2. Treino")
        frame.pack(fill="x", padx=10, pady=6)

        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text="Épocas:").pack(side="left")
        ttk.Entry(row, textvariable=self.epochs_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Batch size:").pack(side="left")
        ttk.Entry(row, textvariable=self.batch_size_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Learning rate:").pack(side="left")
        ttk.Entry(row, textvariable=self.lr_var, width=10).pack(side="left", padx=4)

        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=6, pady=(0, 6))
        self._add_button(btns, "Treinar", self._on_train)

    def _build_infer_section(self) -> None:
        frame = ttk.LabelFrame(self, text="3. Inferência (gerar vídeo trocado)")
        frame.pack(fill="x", padx=10, pady=6)

        row1 = ttk.Frame(frame)
        row1.pack(fill="x", padx=6, pady=4)
        ttk.Label(row1, text="Vídeo de cena:", width=14).pack(side="left")
        ttk.Entry(row1, textvariable=self.infer_scene_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row1, text="Arquivo...", command=self._browse_infer_scene).pack(side="left")

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", padx=6, pady=4)
        ttk.Label(row2, text="Saída:", width=14).pack(side="left")
        ttk.Entry(row2, textvariable=self.infer_output_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row2, text="Salvar como...", command=self._browse_infer_output).pack(side="left")

        row3 = ttk.Frame(frame)
        row3.pack(fill="x", padx=6, pady=4)
        ttk.Label(row3, text="Checkpoint (vazio = mais recente):", width=28).pack(side="left")
        ttk.Entry(row3, textvariable=self.checkpoint_path_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row3, text="Arquivo...", command=self._browse_checkpoint).pack(side="left")

        row4 = ttk.Frame(frame)
        row4.pack(fill="x", padx=6, pady=4)
        ttk.Label(row4, text="Blend:").pack(side="left")
        ttk.Combobox(
            row4, textvariable=self.blend_var, values=["feather", "poisson"], state="readonly", width=10
        ).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(row4, text="Manter áudio original", variable=self.keep_audio_var).pack(side="left")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=6, pady=(0, 6))
        self._add_button(btns, "Gerar vídeo", self._on_infer)

    def _add_button(self, parent, text: str, command) -> None:
        button = ttk.Button(parent, text=text, command=command)
        button.pack(side="left", padx=4)
        self._buttons.append(button)

    # -- browse helpers -------------------------------------------------

    def _browse_infer_scene(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.mov *.avi *.mkv *.webm"), ("Todos", "*.*")])
        if path:
            self.infer_scene_var.set(path)

    def _browse_infer_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if path:
            self.infer_output_var.set(path)

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Checkpoint", "*.pt"), ("Todos", "*.*")])
        if path:
            self.checkpoint_path_var.set(path)

    # -- background task plumbing ---------------------------------------

    def _run_in_thread(self, fn) -> None:
        if self._busy:
            messagebox.showinfo("Ocupado", "Aguarde a tarefa atual terminar.")
            return
        self._set_busy(True)

        def worker():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - surface any error to the log console
                logger.exception("Tarefa falhou: %s", exc)
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in self._buttons:
            button.configure(state=state)

    # -- identity / data_root helpers -------------------------------------

    def _identity_name(self) -> str | None:
        name = self.identity_name_var.get().strip()
        if not name:
            messagebox.showwarning("Identidade", "Defina o nome da identidade na aba 1.")
            return None
        return name

    def _identity_folder(self, name: str) -> Path:
        return self.data_root / "identities" / name

    def _scene_cache_dir(self) -> Path:
        return self.data_root / "scenes" / "_aligned_cache"

    def _checkpoint_dir(self, identity_name: str) -> Path:
        return self.data_root / "checkpoints" / identity_name

    # -- actions ----------------------------------------------------------

    def _on_align_identity(self) -> None:
        name = self._identity_name()
        if name is None:
            return
        resolution = int(self.resolution_var.get())
        identity_folder = self._identity_folder(name)

        def task():
            from align.detector import FaceDetector
            from align.face_cache import build_from_identity
            from datasets.identity_dataset import IdentityDataset

            identity_ds = IdentityDataset(identity_folder)
            detector = FaceDetector()
            saved = build_from_identity(identity_ds, detector, resolution)
            logger.info("Alinhamento da identidade concluído: %d crop(s) salvos.", len(saved))

        self._run_in_thread(task)

    def _on_align_scene(self) -> None:
        dataset_name = self.scene_dataset_var.get()
        scene_root = self.scene_path_var.get().strip()
        if not scene_root:
            messagebox.showwarning("Cena", "Defina o caminho da cena na aba 2.")
            return
        resolution = int(self.resolution_var.get())
        cache_dir = self._scene_cache_dir()

        def task():
            from align.detector import FaceDetector
            from align.face_cache import build_from_scene
            from datasets.scene_dataset import build_scene_dataset

            scene_ds = build_scene_dataset(dataset_name, scene_root)
            detector = FaceDetector()
            saved = build_from_scene(scene_ds, detector, resolution, cache_dir)
            logger.info("Alinhamento da cena concluído: %d crop(s) salvos em %s.", len(saved), cache_dir)

        self._run_in_thread(task)

    def _on_train(self) -> None:
        name = self._identity_name()
        if name is None:
            return
        identity_folder = self._identity_folder(name)
        scene_cache = self._scene_cache_dir()
        resolution = int(self.resolution_var.get())
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            lr = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Treino", "Épocas/batch size/lr inválidos.")
            return

        def task():
            from training.consent import require_identity_consent
            from training.dataset_adapters import IdentityFaceSet, SceneFaceSet
            from training.trainer import Trainer, TrainerConfig
            from datasets.identity_dataset import IdentityDataset

            identity_ds = IdentityDataset(identity_folder)
            consent = require_identity_consent(identity_ds)

            identity_faces = IdentityFaceSet(identity_folder / "_aligned_cache")
            scene_faces = SceneFaceSet(scene_cache)

            config = TrainerConfig(
                resolution=resolution,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                checkpoint_dir=str(self.data_root / "checkpoints"),
            )
            trainer = Trainer(config, identity_name=name)
            trainer.fit(identity_faces, scene_faces, consent)
            logger.info("Treino concluído. Checkpoints em %s/%s/", config.checkpoint_dir, name)

        self._run_in_thread(task)

    def _on_infer(self) -> None:
        name = self._identity_name()
        if name is None:
            return
        scene_video = self.infer_scene_var.get().strip()
        output_path = self.infer_output_var.get().strip()
        if not scene_video or not output_path:
            messagebox.showwarning("Inferência", "Defina o vídeo de cena e o caminho de saída.")
            return

        resolution = int(self.resolution_var.get())
        blend = self.blend_var.get()
        keep_audio = bool(self.keep_audio_var.get())
        explicit_checkpoint = self.checkpoint_path_var.get().strip() or None
        checkpoint_dir = self._checkpoint_dir(name)

        def task():
            from align.detector import FaceDetector
            from inference.swapper import FaceSwapper
            from models.autoencoder import FaceSwapAutoencoder
            from utils.checkpoint import latest_checkpoint, load_checkpoint
            from utils.device import get_device

            checkpoint_path = Path(explicit_checkpoint) if explicit_checkpoint else latest_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"Nenhum checkpoint encontrado em {checkpoint_dir}. Treine primeiro.")

            device = get_device(prefer_cuda=True)
            state = load_checkpoint(checkpoint_path, map_location=str(device))

            model = FaceSwapAutoencoder(
                resolution=state.get("resolution", resolution), latent_dim=state.get("latent_dim", 512)
            )
            model.load_state_dict(state["model_state"])

            detector = FaceDetector()
            swapper = FaceSwapper(
                model, detector, resolution=state.get("resolution", resolution), device=device, blend=blend
            )
            result_path = swapper.process_video(scene_video, output_path, keep_audio=keep_audio)
            logger.info("Vídeo gerado: %s", result_path)

        self._run_in_thread(task)
