"""The second screen: what the run is doing, and how much longer.

The bar is genuinely determinate, which is the whole reason `engine.tiling`
exists — see its module docstring. During the upscale stage it tracks completed
tiles, and during a model download it tracks bytes.

The remaining-time figure starts as the catalog's estimate and crosses over to a
measured rate as tiles complete, weighted so that the first tile — which carries
all of the one-off Vulkan and model-load cost, and would otherwise imply a wildly
pessimistic total — does not dominate. An estimate that visibly lurches is worse
than no estimate, so it is also clamped to never increase by more than a little
between updates.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..engine import runner
from . import metrics as me
from .setup_page import human_time

#: How much weight the measured rate gets once every tile is done. Below that it
#: is blended with the prior in proportion to how much of the run has finished.
_MEASURED_WEIGHT = 0.9


class ProgressPage(QWidget):
    """A determinate progress bar, an ETA, and a way out."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._started = 0.0
        self._stage_started = 0.0
        self._stage = ""
        self._done = 0
        self._total = 0
        self._prior_seconds = 0.0
        self._last_eta: float | None = None

        gu = me.metrics.gu
        outer = QVBoxLayout(self)
        outer.setContentsMargins(gu(2), gu(2), gu(2), gu(2))
        outer.setSpacing(gu(0.75))
        outer.addStretch(1)

        self._title = QLabel("Working…")
        font = self._title.font()
        font.setPointSizeF(font.pointSizeF() * 1.3)
        font.setBold(True)
        self._title.setFont(font)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        outer.addWidget(self._title)

        self._subtitle = QLabel()
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setWordWrap(True)
        outer.addWidget(self._subtitle)

        self._bar = QProgressBar()
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._bar.setMinimumWidth(gu(18))
        self._bar.setTextVisible(True)
        outer.addWidget(self._bar)

        self._detail = QLabel()
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._detail)

        row = QHBoxLayout()
        row.addStretch(1)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        # 500 ms keeps the elapsed counter honest without repainting constantly.
        self._ticker = QTimer(self)
        self._ticker.setInterval(500)
        self._ticker.timeout.connect(self._retick)

    # -- lifecycle --------------------------------------------------------
    def begin(self, job: runner.Job, prior_seconds: float, subtitle: str) -> None:
        self._started = self._stage_started = time.monotonic()
        self._stage = ""
        self._done = self._total = 0
        self._prior_seconds = max(0.0, prior_seconds)
        self._last_eta = None
        self._cancel.setEnabled(True)
        self._cancel.setText("Cancel")
        self._title.setText("Starting…")
        self._subtitle.setText(subtitle)
        self._bar.setRange(0, 0)
        self._detail.setText("")
        self._ticker.start()

    def end(self) -> None:
        self._ticker.stop()

    # -- signals from the worker ------------------------------------------
    def on_stage(self, key: str, text: str) -> None:
        if key != self._stage:
            self._stage = key
            self._stage_started = time.monotonic()
            self._last_eta = None
        self._title.setText(text)
        self._retick()

    def on_progress(self, done: int, total: int) -> None:
        self._done, self._total = done, total
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(done)
        else:
            self._bar.setRange(0, 0)
        self._retick()

    # -- the numbers ------------------------------------------------------
    def _retick(self) -> None:
        elapsed = time.monotonic() - self._started
        parts = [f"Elapsed {human_time(elapsed)}"]

        if self._stage == runner.STAGE_DOWNLOAD and self._total:
            self._bar.setFormat("%p%")
            parts.append(f"{self._done / 1048576:.1f} of "
                         f"{self._total / 1048576:.1f} MB")
        elif self._stage == runner.STAGE_UPSCALE and self._total:
            self._bar.setFormat("%v of %m tiles")
            parts.append(f"Tile {min(self._done + 1, self._total)} of {self._total}")
            eta = self._eta()
            if eta is not None:
                parts.append(f"About {human_time(eta)} left")
        else:
            self._bar.setFormat("%p%")

        self._detail.setText("   ·   ".join(parts))

    def _eta(self) -> float | None:
        """Seconds remaining, blending the prior with the observed tile rate."""
        if not self._total:
            return None
        fraction = self._done / self._total
        prior_left = max(0.0, self._prior_seconds - (time.monotonic() - self._started))
        if self._done <= 0:
            return prior_left or None

        stage_elapsed = time.monotonic() - self._stage_started
        measured_left = (stage_elapsed / self._done) * (self._total - self._done)
        # Trust the measurement more as the run progresses. Early on, one tile's
        # timing is mostly the fixed startup cost and says little about the rest.
        weight = _MEASURED_WEIGHT * fraction
        eta = weight * measured_left + (1 - weight) * prior_left

        # Let it fall freely but rise only slowly, so a momentarily slow tile
        # does not make the number jump backwards.
        if self._last_eta is not None and eta > self._last_eta:
            eta = min(eta, self._last_eta + 1.0)
        self._last_eta = eta
        return eta

    def _on_cancel(self) -> None:
        self._cancel.setEnabled(False)
        self._cancel.setText("Cancelling…")
        self._title.setText("Stopping the run…")
        self.cancel_requested.emit()
