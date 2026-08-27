"""The first screen: pick an image, pick a model, start.

The layout answers the three questions a user has here, in order: which image,
which model, and how long is this going to take. The last one is why the model
combo carries a size and a "best for" line rather than only a name — with
sixteen models whose differences are not guessable from their names, a bare list
would be a worse interface than no choice at all.

Models that have not been downloaded are shown anyway, annotated with their
size. Hiding them behind a separate manager screen would mean the user has to
know the model exists before they can get it; instead, choosing one and pressing
Upscale downloads it as the first stage of the run.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from .. import settings as st
from ..engine import catalog, fetch, runner, tiling
from . import metrics as me

#: What QImageReader can open in practice, as a file dialog filter.
IMAGE_FILTER = ("Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.ppm *.pgm);;"
                "All files (*)")

#: Refuse to start above this many output pixels; the result would not fit in
#: RAM on the machine this targets. 8000x8000 at 4x is already 1 GP.
MAX_OUTPUT_PIXELS = 400_000_000

THUMB_GU = 9


def human_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024:.0f} kB"


def human_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


class SetupPage(QWidget):
    """Choose an image and a model."""

    upscale_requested = Signal()

    def __init__(self, settings: st.Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._source: Path | None = None
        self._source_size: tuple[int, int] = (0, 0)

        gu = me.metrics.gu
        outer = QVBoxLayout(self)
        outer.setContentsMargins(gu(1.5), gu(1.5), gu(1.5), gu(1.5))
        outer.setSpacing(gu(1))

        # -- the image ------------------------------------------------------
        pick = QHBoxLayout()
        pick.setSpacing(gu(1))
        self._thumb = QLabel()
        self._thumb.setFixedSize(gu(THUMB_GU), gu(THUMB_GU))
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setFrameShape(QFrame.Shape.StyledPanel)
        pick.addWidget(self._thumb)

        details = QVBoxLayout()
        details.setSpacing(gu(0.25))
        self._name = QLabel("No image chosen")
        font = self._name.font()
        font.setBold(True)
        self._name.setFont(font)
        self._name.setWordWrap(True)
        self._dims = QLabel("Choose a PNG, JPEG or WebP to get started.")
        self._dims.setWordWrap(True)
        self._open = QPushButton("Open Image…")
        self._open.setDefault(True)
        self._open.clicked.connect(self._choose_file)
        details.addWidget(self._name)
        details.addWidget(self._dims)
        details.addStretch(1)
        row = QHBoxLayout()
        row.addWidget(self._open)
        row.addStretch(1)
        details.addLayout(row)
        pick.addLayout(details, 1)
        outer.addLayout(pick)

        # -- the model ------------------------------------------------------
        box = QGroupBox("Model")
        form = QFormLayout(box)
        form.setSpacing(gu(0.5))
        self._model = QComboBox()
        self._model.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Fixed)
        for m in catalog.MODELS:
            self._model.addItem(m.label, m.id)
        self._model.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model", self._model)

        self._blurb = QLabel()
        self._blurb.setWordWrap(True)
        form.addRow("", self._blurb)

        self._scale = QComboBox()
        self._scale.currentIndexChanged.connect(self._on_scale_changed)
        form.addRow("Scale", self._scale)
        outer.addWidget(box)

        # -- advanced -------------------------------------------------------
        adv = QGroupBox("Advanced")
        adv.setCheckable(True)
        adv.setChecked(False)
        adv_form = QFormLayout(adv)
        adv_form.setSpacing(gu(0.5))

        self._output_scale = QComboBox()
        self._output_scale.currentIndexChanged.connect(self._on_output_scale_changed)
        adv_form.addRow("Output size", self._output_scale)

        self._engine_tile = QComboBox()
        for value in st.ENGINE_TILE_CHOICES:
            self._engine_tile.addItem("Auto" if value == 0 else f"{value} px", value)
        self._engine_tile.setToolTip(
            "How much the GPU processes at once. Lower it if a run fails with an "
            "out-of-memory error; 128 fits comfortably in 4 GB of VRAM.")
        self._engine_tile.currentIndexChanged.connect(self._on_advanced_changed)
        adv_form.addRow("GPU tile", self._engine_tile)

        self._outer_tile = QComboBox()
        for value in st.OUTER_TILE_CHOICES:
            label = ("Auto" if value is None
                     else "Single pass (no progress)" if value == 0 else f"{value} px")
            self._outer_tile.addItem(label, value)
        self._outer_tile.setToolTip(
            "How the image is cut up so progress can be reported. Auto picks a "
            "size from the image. Single pass is a fallback if you ever suspect "
            "a seam.")
        self._outer_tile.currentIndexChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Tiling", self._outer_tile)

        self._device = QComboBox()
        self._device.addItem("Automatic (GPU)", "auto")
        self._device.addItem("CPU only", "cpu")
        self._device.setToolTip("CPU works without a Vulkan driver, but is far slower.")
        self._device.currentIndexChanged.connect(self._on_advanced_changed)
        adv_form.addRow("Device", self._device)

        self._tta = QCheckBox("Higher quality, roughly 8x slower")
        self._tta.setToolTip("Test-time augmentation: upscale eight rotations and "
                             "average them.")
        self._tta.toggled.connect(self._on_advanced_changed)
        adv_form.addRow("TTA", self._tta)
        outer.addWidget(adv)

        outer.addStretch(1)

        # -- go -------------------------------------------------------------
        self._estimate = QLabel()
        self._estimate.setWordWrap(True)
        outer.addWidget(self._estimate)

        go = QHBoxLayout()
        go.addStretch(1)
        self._go = QPushButton("Upscale Image")
        self._go.clicked.connect(self.upscale_requested.emit)
        go.addWidget(self._go)
        outer.addLayout(go)

        self._load_settings()
        self._refresh()

    # -- settings <-> widgets --------------------------------------------
    def _load_settings(self) -> None:
        s = self._settings
        for widget, value in ((self._model, s.model_id),
                              (self._engine_tile, s.engine_tile),
                              (self._outer_tile, s.outer_tile),
                              (self._device, s.device)):
            index = widget.findData(value)
            if index >= 0:
                widget.blockSignals(True)
                widget.setCurrentIndex(index)
                widget.blockSignals(False)
        self._tta.blockSignals(True)
        self._tta.setChecked(s.tta)
        self._tta.blockSignals(False)
        self._rebuild_scales()

    def _on_advanced_changed(self, *_: object) -> None:
        s = self._settings
        s.engine_tile = self._engine_tile.currentData()
        s.outer_tile = self._outer_tile.currentData()
        s.device = self._device.currentData()
        s.tta = self._tta.isChecked()
        self._refresh()

    def _on_model_changed(self, *_: object) -> None:
        self._settings.model_id = self._model.currentData()
        self._rebuild_scales()
        self._refresh()

    def _on_scale_changed(self, *_: object) -> None:
        data = self._scale.currentData()
        if data is not None:
            self._settings.scale = int(data)
            self._rebuild_output_scales()
            self._refresh()

    def _on_output_scale_changed(self, *_: object) -> None:
        self._settings.output_scale = self._output_scale.currentData()
        self._refresh()

    def _rebuild_scales(self) -> None:
        model = self._settings.model()
        self._scale.blockSignals(True)
        self._scale.clear()
        for value in model.scales:
            self._scale.addItem(f"{value}x", value)
        if self._settings.scale not in model.scales:
            self._settings.scale = model.default_scale()
        index = self._scale.findData(self._settings.scale)
        self._scale.setCurrentIndex(max(0, index))
        # A single-scale model has nothing to choose, so the control says so
        # rather than looking broken.
        self._scale.setEnabled(len(model.scales) > 1)
        self._scale.blockSignals(False)
        self._rebuild_output_scales()

    def _rebuild_output_scales(self) -> None:
        """Offer any integer scale up to the engine's, via a final downsample."""
        scale = self._settings.scale
        self._output_scale.blockSignals(True)
        self._output_scale.clear()
        self._output_scale.addItem(f"{scale}x — native", None)
        for value in range(2, scale):
            self._output_scale.addItem(f"{value}x — downsampled from {scale}x", value)
        index = self._output_scale.findData(self._settings.output_scale)
        self._output_scale.setCurrentIndex(max(0, index))
        self._output_scale.setEnabled(self._output_scale.count() > 1)
        self._output_scale.blockSignals(False)

    # -- image ------------------------------------------------------------
    def _choose_file(self) -> None:
        start = self._settings.last_open_dir or str(Path.home())
        name, _ = QFileDialog.getOpenFileName(self, "Open Image", start, IMAGE_FILTER)
        if name:
            self.load_image(Path(name))

    def load_image(self, path: Path) -> bool:
        image = QImage(str(path))
        if image.isNull():
            self._name.setText("Could not read that file")
            self._dims.setText(f"{path.name} is not an image this app can open.")
            self._thumb.clear()
            self._source = None
            self._refresh()
            return False
        self._source = path
        self._source_size = (image.width(), image.height())
        self._settings.last_open_dir = str(path.parent)
        gu = me.metrics.gu(THUMB_GU)
        self._thumb.setPixmap(QPixmap.fromImage(image).scaled(
            gu, gu, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self._name.setText(path.name)
        self._refresh()
        return True

    @property
    def source(self) -> Path | None:
        return self._source

    @property
    def source_size(self) -> tuple[int, int]:
        return self._source_size

    def outer_tile_for_source(self) -> int:
        return runner.plan_tile_size(*self._source_size, self._settings.outer_tile)

    def build_job(self) -> runner.Job:
        s = self._settings
        return runner.Job(
            source=self._source, model=s.model(), scale=s.scale,
            output_scale=s.output_scale, engine_tile=s.engine_tile,
            outer_tile=self.outer_tile_for_source(), ctx=tiling.DEFAULT_CONTEXT,
            tta=s.tta, gpu=s.gpu_arg(), binary_path=s.binary_path or None)

    # -- the summary line -------------------------------------------------
    def _refresh(self) -> None:
        model = self._settings.model()
        scale = self._settings.scale
        pending = fetch.download_size(model, scale)
        licence = f" · {model.licence}" if model.licence else ""
        self._blurb.setText(f"{model.blurb}\n{model.author}{licence}")

        if self._source is None:
            self._go.setEnabled(False)
            self._dims.setText("Choose a PNG, JPEG or WebP to get started.")
            self._estimate.setText(
                f"{model.label} needs a {human_bytes(pending)} download."
                if pending else f"{model.label} is downloaded and ready.")
            return

        width, height = self._source_size
        out_scale = self._settings.output_scale or scale
        out = (width * out_scale, height * out_scale)
        self._dims.setText(f"{width} x {height}  ->  {out[0]} x {out[1]}")

        too_big = (width * scale) * (height * scale) > MAX_OUTPUT_PIXELS
        self._go.setEnabled(not too_big)
        if too_big:
            self._estimate.setText(
                f"That would produce {(width * scale) * (height * scale) / 1e6:.0f} "
                f"megapixels, which will not fit in memory. Try a smaller scale.")
            return

        rate = self._settings.calibration.get(model.id, scale, self._settings.device)
        seconds = runner.estimate_seconds(model, width, height, rate)
        if self._settings.tta:
            seconds *= 8
        tiles = len(tiling.plan_tiles(width, height, self.outer_tile_for_source(),
                                      tiling.DEFAULT_CONTEXT))
        basis = "measured on this machine" if rate else "estimated"
        bits = [f"About {human_time(seconds)} ({basis}), in {tiles} tile"
                f"{'s' if tiles != 1 else ''}."]
        if pending:
            bits.append(f"Downloads {human_bytes(pending)} first.")
        raw = tiling.output_bytes(width, height, out_scale)
        if raw > 512 * 1024 ** 2:
            bits.append(f"The result is about {human_bytes(raw)} in memory.")
        self._estimate.setText(" ".join(bits))

    def refresh(self) -> None:
        """Re-read state that may have changed elsewhere (a finished download)."""
        self._refresh()
