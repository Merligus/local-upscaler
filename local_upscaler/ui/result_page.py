"""The third screen: original, upscaled, and the wipe between them.

The three views are a segmented control over a single `ImageView`, not three
widgets, so zoom and position survive switching between them — the comparison is
only useful if "the same place" means the same place.

Saving is offered but never automatic. The result lives in memory until asked
for, and the default filename records which model produced it, because after
trying three models the files are otherwise indistinguishable.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QToolButton,
                               QVBoxLayout, QWidget)

from .. import settings as st
from . import metrics as me
from .image_view import MODE_COMPARE, MODE_ORIGINAL, MODE_UPSCALED, ImageView
from .setup_page import human_time

SAVE_FILTER = ("PNG image (*.png);;JPEG image (*.jpg *.jpeg);;"
               "WebP image (*.webp);;All files (*)")


class ResultPage(QWidget):
    """Shows the finished upscale and offers to save it."""

    back_requested = Signal()

    def __init__(self, settings: st.Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._source: Path | None = None
        self._model_id = ""
        self._scale = 4

        gu = me.metrics.gu
        outer = QVBoxLayout(self)
        outer.setContentsMargins(gu(0.75), gu(0.75), gu(0.75), gu(0.75))
        outer.setSpacing(gu(0.5))

        # -- the segmented control ------------------------------------------
        bar = QHBoxLayout()
        bar.setSpacing(gu(0.25))
        self._modes = QButtonGroup(self)
        self._modes.setExclusive(True)
        for index, (mode, label) in enumerate(((MODE_ORIGINAL, "Original"),
                                               (MODE_UPSCALED, "Upscaled"),
                                               (MODE_COMPARE, "Compare"))):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setChecked(mode == MODE_COMPARE)
            self._modes.addButton(button, index)
            button.clicked.connect(lambda _checked, m=mode: self._set_mode(m))
            bar.addWidget(button)
        bar.addSpacing(gu(1))

        self._filter = QComboBox()
        self._filter.addItem("Original: real pixels", "nearest")
        self._filter.addItem("Original: smoothed", "smooth")
        self._filter.setToolTip(
            "How the original is magnified for the comparison. Real pixels shows "
            "the source as it actually is; smoothing it compares the model "
            "against an interpolation instead, which flatters the model.")
        self._filter.currentIndexChanged.connect(self._on_filter_changed)
        bar.addWidget(self._filter)

        bar.addStretch(1)
        self._zoom_label = QLabel()
        bar.addWidget(self._zoom_label)
        for text, tip, slot in (("Fit", "Fit the whole image (0)", self._fit),
                                ("100%", "One output pixel per screen pixel (1)",
                                 self._actual)):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        outer.addLayout(bar)

        # -- the canvas -----------------------------------------------------
        self._view = ImageView()
        self._view.zoom_changed.connect(self._on_zoom_changed)
        outer.addWidget(self._view, 1)

        # -- the footer -----------------------------------------------------
        footer = QHBoxLayout()
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        footer.addWidget(self._summary, 1)
        self._back = QPushButton("Upscale Another")
        self._back.setToolTip("Back to the first screen, keeping this image loaded.")
        self._back.clicked.connect(self.back_requested.emit)
        footer.addWidget(self._back)
        self._save = QPushButton("Save Image…")
        self._save.setDefault(True)
        self._save.clicked.connect(self._save_as)
        footer.addWidget(self._save)
        outer.addLayout(footer)

        index = self._filter.findData(settings.compare_filter)
        if index >= 0:
            self._filter.setCurrentIndex(index)
        self._view.set_filter(settings.compare_filter)
        self._view.set_mode(MODE_COMPARE)

    # -- content ----------------------------------------------------------
    def show_result(self, source: Path, original: QImage, upscaled: QImage,
                    model_label: str, model_id: str, scale: int,
                    elapsed: float, tiles: int) -> None:
        self._source = source
        self._model_id = model_id
        self._scale = scale
        self._view.set_images(original, upscaled)
        self._summary.setText(
            f"{original.width()} x {original.height()}  ->  "
            f"{upscaled.width()} x {upscaled.height()}   ·   {model_label}   ·   "
            f"{human_time(elapsed)} in {tiles} tile{'s' if tiles != 1 else ''}")
        self._on_zoom_changed(self._view.zoom())

    # -- view controls ----------------------------------------------------
    def _set_mode(self, mode: str) -> None:
        self._view.set_mode(mode)
        self._filter.setEnabled(mode in (MODE_COMPARE, MODE_ORIGINAL))

    def _on_filter_changed(self) -> None:
        name = self._filter.currentData()
        self._settings.compare_filter = name
        self._view.set_filter(name)

    def _fit(self) -> None:
        self._view.fit()

    def _actual(self) -> None:
        self._view.zoom_to_actual()

    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_label.setText(f"{zoom * 100:.0f}%")

    # -- saving -----------------------------------------------------------
    def _default_name(self) -> str:
        stem = self._source.stem if self._source else "upscaled"
        return f"{stem}_{self._model_id}_x{self._scale}.png"

    def _save_as(self) -> None:
        image = self._view.upscaled_image()
        if image is None:
            return
        start = (self._settings.last_save_dir
                 or (str(self._source.parent) if self._source else str(Path.home())))
        target, _ = QFileDialog.getSaveFileName(
            self, "Save Upscaled Image", str(Path(start) / self._default_name()),
            SAVE_FILTER)
        if not target:
            return
        path = Path(target)
        if not path.suffix:
            path = path.with_suffix(".png")
        # JPEG has no alpha; saving an RGBA image to it silently loses the mask
        # or fails outright depending on the plugin, so flatten deliberately.
        to_save = image
        if path.suffix.lower() in (".jpg", ".jpeg") and image.hasAlphaChannel():
            to_save = image.convertToFormat(QImage.Format.Format_RGB32)
        if to_save.save(str(path), quality=95):
            self._settings.last_save_dir = str(path.parent)
            self._summary.setText(f"Saved to {path}")
        else:
            QMessageBox.warning(self, "Could not save",
                                f"Writing {path} failed. Check the folder is "
                                f"writable and has room for the file.")
