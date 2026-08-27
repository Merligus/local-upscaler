"""Finding and invoking the `realesrgan-ncnn-vulkan` executable.

The engine is a standalone ~11 MB Vulkan binary, not a Python library. That is a
deliberate choice and worth recording, because the obvious alternative does not
work on this hardware:

    The development machine has a GTX 1050 Ti — Pascal, compute capability
    sm_61. CUDA 13.0 removed offline compilation for everything below sm_75, and
    Arch's `python-pytorch-cuda` is built against CUDA 13.3, so it contains no
    kernels this GPU can run. PyTorch here would silently be CPU-only. ncnn goes
    through Vulkan instead, which Pascal supports perfectly well, so the 4 GB
    card is actually usable.

Three places are searched, in order: a path the user pinned in settings, then
`PATH` (which covers `aur/realesrgan-ncnn-vulkan-bin` and `aur/upscayl-ncnn`),
then the copy `--fetch-engine` manages under `paths.engine_dir()`. Having both a
system and a managed source means an AUR package that stops working does not
leave the app dead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .. import paths

#: Executable names to look for on PATH. `realesrgan-ncnn-vulkan` is upstream's
#: and what `aur/realesrgan-ncnn-vulkan-bin` installs; `upscayl-bin` is the
#: actively maintained Upscayl fork from `aur/upscayl-ncnn`, which takes the
#: same arguments.
CANDIDATE_NAMES = ("realesrgan-ncnn-vulkan", "upscayl-bin")

#: Official upstream Linux build. The checksum is the one `aur/realesrgan-ncnn-
#: vulkan-bin` pins, independently verified against the download.
ENGINE_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
              "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip")
ENGINE_SHA256 = "e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96"
ENGINE_ZIP_BYTES = 46931474
#: The only member worth extracting; the rest of the zip is demo media and a
#: handful of models the catalog already covers.
ENGINE_MEMBER = "realesrgan-ncnn-vulkan"


def managed_path() -> Path:
    """Where `--fetch-engine` puts its copy."""
    return paths.engine_dir() / ENGINE_MEMBER


def find(explicit: str | None = None) -> Path | None:
    """Locate a usable engine binary, or None.

    `explicit` is the settings override and wins outright when it is set and
    executable — a user who pinned a path meant it.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
    for name in CANDIDATE_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    managed = managed_path()
    if managed.is_file() and os.access(managed, os.X_OK):
        return managed
    return None


def child_env() -> dict[str, str]:
    """Environment for the engine subprocess.

    Vulkan overlay layers inject themselves into every Vulkan process, including
    this one. MangoHud is installed on the development machine and does exactly
    that: it prints its own startup banner into the engine's stderr, which is the
    same stream the progress parser reads, and costs a little time for an overlay
    nobody will ever see on a headless compute job. Turned off explicitly rather
    than worked around in the parser.
    """
    env = dict(os.environ)
    env["MANGOHUD"] = "0"
    env["DISABLE_VKBASALT"] = "1"
    return env


def probe(path: Path, timeout: float = 20.0) -> str | None:
    """Return the binary's usage text if it runs and looks right, else None.

    `-h` exits non-zero upstream, so the return code is deliberately ignored;
    what is checked is that the usage text contains the flags this app relies on.
    """
    try:
        r = subprocess.run([str(path), "-h"], capture_output=True, text=True,
                           timeout=timeout, env=child_env())
    except (OSError, subprocess.SubprocessError):
        return None
    text = (r.stderr or "") + (r.stdout or "")
    needed = ("-i input-path", "-m model-path", "-n model-name", "-t tile-size")
    return text if all(flag in text for flag in needed) else None


def build_argv(binary: Path, in_path: Path, out_path: Path, model_id: str,
               scale: int, models_dir: Path, engine_tile: int = 128,
               gpu: int | None = None, tta: bool = False,
               verbose: bool = True) -> list[str]:
    """The command line for one engine run.

    `in_path`/`out_path` may be files or directories; directory mode is what
    makes the progress bar work (see `runner`).

    `models_dir` must have `models` as a path component. The binary derives its
    internal `prepadding` from that substring and refuses to start without it,
    printing `unknown model dir type`. `paths.models_dir()` guarantees it.
    """
    argv = [
        str(binary),
        "-i", str(in_path),
        "-o", str(out_path),
        "-n", model_id,
        "-m", str(models_dir),
        "-s", str(scale),
        "-t", str(engine_tile),
        "-f", "png",
    ]
    if gpu is not None:
        # -1 selects CPU. Slow, but it keeps the app usable with no working
        # Vulkan device rather than failing outright.
        argv += ["-g", str(gpu)]
    if tta:
        argv.append("-x")
    if verbose:
        argv.append("-v")
    return argv
