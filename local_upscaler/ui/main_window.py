"""The window, and the three-screen flow through it.

    Setup  --Upscale-->  Progress  --done-->  Result
      ^                     |                   |
      +---------------------+-------------------+
             cancel / failure        Upscale Another

A `QStackedWidget` rather than dialogs, because the flow is linear and a modal
progress dialog over a window with nothing in it is just a worse version of this.

Two conversions deserve a note, since both are places where a large image could
quietly be duplicated in memory:

* The result arrives from the engine as a PIL image and has to become a
  `QImage`. `QImage` requires its buffer to outlive it and does not take
  ownership, so the bytes object is kept alive on the widget alongside the
  image. Dropping it produces a window full of garbage or a crash, depending on
  what the allocator does with the freed page.
* The *original* is loaded straight from the file with `QImage`, rather than
  converted from the PIL copy the runner already made, so only one full-size
  copy of each image is ever alive.

Qt conventions follow soundboard's: no `setStyleSheet`, no `setStyle`, every
size derived from `ui.metrics`, and icons through `ui.icons` so a theme change
repaints them.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget

from .. import settings as st
from ..engine import binary, runner
from . import metrics as me
from .progress_page import ProgressPage
from .result_page import ResultPage
from .setup_page import SetupPage
from .worker import UpscaleWorker

PAGE_SETUP, PAGE_PROGRESS, PAGE_RESULT = 0, 1, 2


def pil_to_qimage(image: Image.Image) -> tuple[QImage, bytes]:
    """Convert without an intermediate copy, returning the buffer to keep alive.

    The caller **must** hold on to the returned bytes for as long as the QImage
    is used: `QImage` wraps the pointer it is given and does not copy it.
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    fmt = (QImage.Format.Format_RGBA8888 if image.mode == "RGBA"
           else QImage.Format.Format_RGB888)
    data = image.tobytes()
    qimage = QImage(data, image.width, image.height,
                    image.width * len(image.mode), fmt)
    return qimage, data


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Local Upscaler")
        self._settings = st.load()
        self._thread: QThread | None = None
        self._worker: UpscaleWorker | None = None
        #: Keeps the QImage buffers alive. See `pil_to_qimage`.
        self._buffers: list[bytes] = []

        self._stack = QStackedWidget()
        self._setup = SetupPage(self._settings)
        self._progress = ProgressPage()
        self._result = ResultPage(self._settings)
        for page in (self._setup, self._progress, self._result):
            self._stack.addWidget(page)
        self.setCentralWidget(self._stack)

        self._setup.upscale_requested.connect(self._start)
        self._progress.cancel_requested.connect(self._cancel)
        self._result.back_requested.connect(self._back_to_setup)

        gu = me.metrics.gu
        self.resize(gu(38), gu(30))
        self.setMinimumSize(gu(24), gu(20))
        self._centre()

    def _centre(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        geometry = self.frameGeometry()
        geometry.moveCenter(available.center())
        self.move(geometry.topLeft())

    # -- running ----------------------------------------------------------
    def _start(self) -> None:
        if self._setup.source is None or self._thread is not None:
            return
        if binary.find(self._settings.binary_path or None) is None:
            QMessageBox.warning(
                self, "No upscaling engine",
                "The upscaling engine is not installed.\n\n"
                "Fetch a copy with:\n"
                "    python3 -m local_upscaler --fetch-engine\n\n"
                "or install it system-wide with:\n"
                "    paru -S realesrgan-ncnn-vulkan-bin")
            return

        job = self._setup.build_job()
        st.save(self._settings)

        width, height = self._setup.source_size
        rate = self._settings.calibration.get(job.model.id, job.scale)
        prior = runner.estimate_seconds(job.model, width, height, rate)
        if job.tta:
            prior *= 8
        self._progress.begin(
            job, prior,
            f"{job.source.name}  ·  {width} x {height}  ·  {job.model.label}")
        self._stack.setCurrentIndex(PAGE_PROGRESS)

        self._thread = QThread(self)
        self._worker = UpscaleWorker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.stage.connect(self._progress.on_stage)
        self._worker.progress.connect(self._progress.on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _teardown(self) -> None:
        """Stop the thread and drop both objects. Safe to call more than once."""
        self._progress.end()
        thread, self._thread = self._thread, None
        worker, self._worker = self._worker, None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

    # -- outcomes ---------------------------------------------------------
    def _on_finished(self, result: runner.Result) -> None:
        job = self._setup.build_job()
        self._teardown()

        self._settings.calibration.record(job.model.id, job.scale, result.sec_per_mpx)
        st.save(self._settings)

        upscaled, buffer = pil_to_qimage(result.image)
        original = QImage(str(job.source))
        # Release the previous run's buffers only now that the new ones exist,
        # so switching models never shows a blank canvas.
        self._buffers = [buffer]
        self._result.show_result(
            source=job.source, original=original, upscaled=upscaled,
            model_label=job.model.label, model_id=job.model.id,
            scale=job.effective_output_scale(), elapsed=result.elapsed,
            tiles=result.tiles)
        self._stack.setCurrentIndex(PAGE_RESULT)
        self._setup.refresh()

    def _on_failed(self, message: str) -> None:
        self._teardown()
        self._stack.setCurrentIndex(PAGE_SETUP)
        QMessageBox.warning(self, "Upscaling failed", message)

    def _on_cancelled(self) -> None:
        self._teardown()
        self._stack.setCurrentIndex(PAGE_SETUP)

    def _back_to_setup(self) -> None:
        self._stack.setCurrentIndex(PAGE_SETUP)
        self._setup.refresh()

    # -- window -----------------------------------------------------------
    def open_path(self, path: Path) -> bool:
        """Load an image given on the command line."""
        return self._setup.load_image(path)

    def closeEvent(self, event) -> None:
        if self._thread is not None:
            self._cancel()
            self._teardown()
        st.save(self._settings)
        super().closeEvent(event)
