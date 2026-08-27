"""Running one upscale, start to finish, with honest progress.

Deliberately free of Qt. The UI wraps this in a worker thread and forwards its
callbacks to signals; keeping the orchestration plain Python means it can be
driven from `--bench` and from tests without a display.

The shape of a run:

    fetch the model if absent  ->  cut the source into tiles  ->  one engine
    subprocess over the whole tile directory  ->  crop and paste the tiles back
    ->  optionally downsample to a smaller requested scale

The single subprocess is the important part. The engine spends 2-13 s on Vulkan
init, shader compilation and model load before it does any work at all (see the
measurements in `catalog`), so per-tile invocation would multiply that fixed
cost by the tile count. Directory mode pays it once, and as a bonus the engine's
own load/proc/save threads overlap disk I/O with GPU work — measured 26% *faster*
than handing it the whole image in one piece, not slower.

Progress comes from `-v`, which makes the engine print one `<in> -> <out> done`
line per finished file. Everything else on that stream is banner noise (the GPU
capability dump alone is over a hundred lines) and is ignored by pattern rather
than by position.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .. import paths
from . import binary, fetch, tiling
from .catalog import Model

#: `on_stage(key, human_text)`.
StageCb = Callable[[str, str], None]
#: `on_progress(done, total)` within the current stage. `total` 0 means unknown.
ProgressCb = Callable[[int, int], None]

STAGE_DOWNLOAD = "download"
STAGE_PREPARE = "prepare"
STAGE_UPSCALE = "upscale"
STAGE_ASSEMBLE = "assemble"

#: One finished file. Anchored on `done` at end of line so a path that happens
#: to contain the word cannot produce a phantom tick.
_DONE_RE = re.compile(r"->.*\bdone\s*$")

#: Substrings that identify a failure worth explaining rather than dumping.
_HINTS = (
    ("vkallocatememory", "The GPU ran out of memory. Lower the engine tile size "
                         "in Advanced (try 64), or switch Device to CPU."),
    ("out of memory", "The GPU ran out of memory. Lower the engine tile size "
                      "in Advanced (try 64), or switch Device to CPU."),
    ("unknown model dir type", "The models directory must have 'models' as a path "
                               "component; the engine refuses anything else."),
    ("failed to allocate", "The GPU ran out of memory. Lower the engine tile size "
                           "in Advanced, or switch Device to CPU."),
    ("no vulkan device", "No Vulkan device was found. Switch Device to CPU in "
                         "Advanced, or check that your GPU driver is installed."),
    ("vkcreateinstance", "Vulkan could not start. Switch Device to CPU in Advanced."),
)


class UpscaleError(Exception):
    """The engine could not complete the run."""


class Cancelled(Exception):
    """The user stopped the run."""


@dataclass
class Job:
    """Everything one run needs to know."""

    source: Path
    model: Model
    #: What the engine is asked for. Must be one of `model.scales`.
    scale: int
    #: What the user wants. When smaller than `scale`, the result is Lanczos
    #: downsampled at the end — a 4x model downsampled to 2x beats any 2x model
    #: available here, and it is the only way to get 2x out of most of them.
    output_scale: int | None = None
    #: Engine's internal tile, `-t`. 128 is comfortable inside 4 GB of VRAM;
    #: 512 was measured to exhaust it on the RRDBNet models.
    engine_tile: int = 128
    #: Outer tile for progress granularity. 0 means one single pass.
    outer_tile: int = 0
    ctx: int = tiling.DEFAULT_CONTEXT
    tta: bool = False
    #: `-g`. None lets the engine choose; -1 forces CPU.
    gpu: int | None = None
    binary_path: str | None = None

    def effective_output_scale(self) -> int:
        return self.output_scale or self.scale


@dataclass
class Result:
    image: Image.Image
    elapsed: float
    #: Wall seconds per megapixel of *input*, for `settings.Calibration`.
    sec_per_mpx: float
    tiles: int
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    stderr_tail: str = field(default="", repr=False)


def _explain(stderr: str, returncode: int) -> str:
    low = stderr.lower()
    for needle, hint in _HINTS:
        if needle in low:
            return hint
    tail = [ln for ln in stderr.strip().splitlines() if ln.strip()][-4:]
    return (f"The upscaling engine exited with code {returncode}."
            + ("\n" + "\n".join(tail) if tail else ""))


class Runner:
    """Runs one `Job`. Not reusable; make a new one per run."""

    def __init__(self, job: Job, on_stage: StageCb | None = None,
                 on_progress: ProgressCb | None = None) -> None:
        self.job = job
        self._on_stage = on_stage
        self._on_progress = on_progress
        self._cancelled = False
        self._proc: subprocess.Popen | None = None

    # -- control ----------------------------------------------------------
    def cancel(self) -> None:
        """Ask the run to stop. Safe to call from another thread."""
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _check(self) -> None:
        if self._cancelled:
            raise Cancelled()

    def _stage(self, key: str, text: str) -> None:
        if self._on_stage is not None:
            self._on_stage(key, text)

    def _progress(self, done: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total)

    # -- the run ----------------------------------------------------------
    def run(self) -> Result:
        job = self.job
        exe = binary.find(job.binary_path)
        if exe is None:
            raise UpscaleError(
                "No upscaling engine found. Install one with\n"
                "    paru -S realesrgan-ncnn-vulkan-bin\n"
                "or let the app fetch its own copy:\n"
                "    python3 -m local_upscaler --fetch-engine")

        if fetch.missing(job.model, job.scale):
            self._stage(STAGE_DOWNLOAD, f"Downloading {job.model.label}…")
            try:
                fetch.fetch_model(
                    job.model, job.scale,
                    progress=lambda d, t, _l: self._progress(d, t),
                    is_cancelled=lambda: self._cancelled)
            except fetch.Cancelled as e:
                raise Cancelled() from e
        self._check()

        self._stage(STAGE_PREPARE, "Reading the image…")
        try:
            with Image.open(job.source) as im:
                im.load()
                mode = "RGBA" if im.mode in ("RGBA", "LA", "PA") else "RGB"
                source = im.convert(mode)
        except (OSError, ValueError) as e:
            raise UpscaleError(f"Could not read {job.source.name}: {e}") from e

        width, height = source.size
        work = paths.work_dir() / f"run-{int(time.time() * 1000)}"
        started = time.monotonic()
        try:
            tiles = self._prepare(source, work, width, height)
            self._check()
            self._run_engine(exe, work, len(tiles))
            self._check()
            self._stage(STAGE_ASSEMBLE, "Putting the tiles back together…")
            out = self._assemble(tiles, work, (width, height), mode)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        out = self._postscale(out)
        elapsed = time.monotonic() - started
        mpx = (width * height) / 1e6
        return Result(image=out, elapsed=elapsed,
                      sec_per_mpx=elapsed / mpx if mpx else 0.0,
                      tiles=len(tiles), source_size=(width, height),
                      output_size=out.size)

    # -- steps ------------------------------------------------------------
    def _prepare(self, source: Image.Image, work: Path,
                 width: int, height: int) -> list[tiling.Tile]:
        """Cut the source into tiles on disk. Returns the plan."""
        job = self.job
        tile = job.outer_tile or max(width, height)      # 0 => a single tile
        ctx = 0 if job.outer_tile == 0 else job.ctx
        tiles = tiling.plan_tiles(width, height, tile, ctx)
        self._stage(STAGE_PREPARE,
                    f"Preparing {len(tiles)} tile{'s' if len(tiles) != 1 else ''}…")
        self._progress(0, len(tiles))
        (work / "out").mkdir(parents=True, exist_ok=True)
        tiling.split(source, tiles, work / "in")
        return tiles

    def _run_engine(self, exe: Path, work: Path, total: int) -> None:
        """One subprocess over the tile directory, counting `done` lines."""
        job = self.job
        argv = binary.build_argv(
            exe, work / "in", work / "out", job.model.id, job.scale,
            paths.models_dir(), engine_tile=job.engine_tile, gpu=job.gpu,
            tta=job.tta, verbose=True)

        self._stage(STAGE_UPSCALE, f"Upscaling with {job.model.label}…")
        self._progress(0, total)
        done = 0
        noise: list[str] = []
        try:
            self._proc = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=binary.child_env())
        except OSError as e:
            raise UpscaleError(f"Could not start {exe}: {e}") from e

        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            if _DONE_RE.search(line.strip()):
                done += 1
                self._progress(done, total)
            elif line.strip():
                noise.append(line.rstrip())
        code = self._proc.wait()
        self._proc = None

        if self._cancelled:
            raise Cancelled()
        if code != 0:
            raise UpscaleError(_explain("\n".join(noise), code))
        if done < total:
            # The engine exited cleanly having written fewer files than it was
            # given. Assembly would fail on a missing tile with a worse message.
            raise UpscaleError(
                f"The engine finished {done} of {total} tiles. "
                f"The run is incomplete; try a smaller engine tile size.")

    def _assemble(self, tiles: list[tiling.Tile], work: Path,
                  size: tuple[int, int], mode: str) -> Image.Image:
        try:
            return tiling.reassemble(tiles, work / "out", size, self.job.scale, mode)
        except (OSError, ValueError) as e:
            raise UpscaleError(f"Could not reassemble the tiles: {e}") from e

    def _postscale(self, image: Image.Image) -> Image.Image:
        """Downsample when the user asked for less than the model's native scale."""
        want = self.job.effective_output_scale()
        if want == self.job.scale:
            return image
        w, h = image.size
        factor = want / self.job.scale
        target = (max(1, round(w * factor)), max(1, round(h * factor)))
        self._stage(STAGE_ASSEMBLE, f"Resampling to {want}x…")
        return image.resize(target, Image.LANCZOS)


def plan_tile_size(width: int, height: int, override: int | None = None) -> int:
    """Outer tile size for an image, honouring a settings override.

    `override` of 0 means the user asked for a single pass; None means auto.
    """
    if override is not None:
        return max(0, override)
    return tiling.auto_tile_size(width, height)


def estimate_seconds(model: Model, width: int, height: int,
                     sec_per_mpx: float | None = None) -> float:
    """How long a run should take, for the pre-run estimate and the initial ETA."""
    mpx = (width * height) / 1e6
    rate = sec_per_mpx if sec_per_mpx and sec_per_mpx > 0 else model.sec_per_mpx
    return model.startup_s + rate * mpx
