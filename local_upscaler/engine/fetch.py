"""Downloading models and, if needed, the engine binary.

Nothing is bundled with this repository: models are ~1-64 MB each and 431 MB in
total across the catalog, and the engine is an 11 MB prebuilt executable. Both
are fetched on demand into XDG data directories, so a checkout stays small and
the user only pays for the models they actually pick.

Two rules everything here follows:

* **Verify before installing.** A download that finishes is not a download that
  succeeded — a captive portal, a rate limit or a moved URL all return HTTP 200
  with an HTML body, and writing that to `ultrasharp-4x.bin` produces a model
  file that fails much later with an incomprehensible ncnn error. Model files
  are checked against the exact byte counts in `catalog`, and the engine zip
  against a SHA-256. A mismatch raises here, where the message can be useful.
* **Install atomically.** Everything downloads to a `.part` file beside the
  destination and is `os.replace`d into place only after it verifies, so an
  interrupted fetch can never leave a half-written model that looks present to
  `have_model`.

`urllib` rather than `requests`: this app's dependency policy is "system
packages only, no venv" (see docs/COMPATIBILITY.md), and the standard library
is entirely adequate for a straight GET.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from .. import paths
from . import binary
from .catalog import Model

#: `progress(done_bytes, total_bytes, label)`. `total` is 0 when unknown.
ProgressCb = Callable[[int, int, str], None]
#: Returns True to abort an in-flight download.
CancelCb = Callable[[], bool]

_CHUNK = 256 * 1024
_TIMEOUT = 60


class FetchError(Exception):
    """A download failed, or arrived corrupt."""


class Cancelled(Exception):
    """The user cancelled an in-flight download."""


def _download(url: str, dest: Path, expect_bytes: int | None = None,
              sha256: str | None = None, progress: ProgressCb | None = None,
              is_cancelled: CancelCb | None = None, label: str = "",
              base: int = 0, grand_total: int = 0) -> None:
    """Fetch `url` to `dest`, verifying size and/or checksum before installing.

    `base`/`grand_total` let a multi-file fetch report one continuous bar across
    all of its files rather than restarting at zero for each.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    got = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "local-upscaler"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp, part.open("wb") as f:
            declared = int(resp.headers.get("Content-Length") or 0)
            total = grand_total or expect_bytes or declared
            while True:
                if is_cancelled is not None and is_cancelled():
                    raise Cancelled(label or url)
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                got += len(chunk)
                if progress is not None:
                    progress(base + got, total, label)
    except urllib.error.URLError as e:
        part.unlink(missing_ok=True)
        raise FetchError(f"could not download {url}: {e.reason}") from e
    except OSError as e:
        part.unlink(missing_ok=True)
        raise FetchError(f"could not write {dest}: {e}") from e
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    if expect_bytes is not None and got != expect_bytes:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"{dest.name}: expected {expect_bytes} bytes, got {got}. "
            f"The download was truncated, or the URL now serves something else.")
    if sha256 is not None and digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise FetchError(f"{dest.name}: checksum mismatch — the file is not what it should be.")
    os.replace(part, dest)


# ---------------------------------------------------------------- models
def model_paths(model: Model, scale: int) -> list[Path]:
    """The files `model` needs on disk to run at `scale`."""
    d = paths.models_dir()
    return [d / name for name in model.filenames(scale)]


def missing(model: Model, scale: int) -> list[Path]:
    """Which of `model`'s files are absent or the wrong size.

    Size is checked, not just existence, so a fetch interrupted before this
    module existed — or a file truncated by a full disk — is re-downloaded
    rather than handed to the engine.
    """
    expected = dict(zip(model.filenames(scale), (model.param_bytes, model.bin_bytes)))
    out = []
    for p in model_paths(model, scale):
        try:
            if p.stat().st_size != expected[p.name]:
                out.append(p)
        except OSError:
            out.append(p)
    return out


def have_model(model: Model, scale: int) -> bool:
    return not missing(model, scale)


def download_size(model: Model, scale: int) -> int:
    """Bytes still to fetch for `model` at `scale`. Zero when it is ready."""
    expected = dict(zip(model.filenames(scale), (model.param_bytes, model.bin_bytes)))
    return sum(expected[p.name] for p in missing(model, scale))


def fetch_model(model: Model, scale: int, progress: ProgressCb | None = None,
                is_cancelled: CancelCb | None = None) -> None:
    """Download whatever `model` is missing at `scale`. A no-op when complete."""
    todo = missing(model, scale)
    if not todo:
        return
    expected = dict(zip(model.filenames(scale), (model.param_bytes, model.bin_bytes)))
    total = sum(expected[p.name] for p in todo)
    base = 0
    for path in todo:
        _download(model.base_url + path.name, path, expect_bytes=expected[path.name],
                  progress=progress, is_cancelled=is_cancelled,
                  label=f"Downloading {model.label}", base=base, grand_total=total)
        base += expected[path.name]


# ---------------------------------------------------------------- engine
def fetch_engine(progress: ProgressCb | None = None,
                 is_cancelled: CancelCb | None = None) -> Path:
    """Download the upstream release and install just the executable.

    The release zip is 47 MB, of which 11 MB is the binary and the rest is demo
    media and a few models the catalog already covers, so only the one member is
    extracted.
    """
    dest = binary.managed_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = paths.cache_dir() / "engine.zip"
    _download(binary.ENGINE_URL, archive, expect_bytes=binary.ENGINE_ZIP_BYTES,
              sha256=binary.ENGINE_SHA256, progress=progress,
              is_cancelled=is_cancelled, label="Downloading the upscaling engine")

    part = dest.with_name(dest.name + ".part")
    try:
        with zipfile.ZipFile(archive) as z:
            member = next((n for n in z.namelist()
                           if Path(n).name == binary.ENGINE_MEMBER and not n.endswith("/")), None)
            if member is None:
                raise FetchError(f"{archive.name} does not contain {binary.ENGINE_MEMBER}")
            with z.open(member) as src, part.open("wb") as f:
                while chunk := src.read(_CHUNK):
                    f.write(chunk)
    except (zipfile.BadZipFile, OSError) as e:
        part.unlink(missing_ok=True)
        raise FetchError(f"could not unpack {archive}: {e}") from e

    part.chmod(0o755)
    os.replace(part, dest)
    archive.unlink(missing_ok=True)     # 47 MB of cache with nothing left to give
    return dest
