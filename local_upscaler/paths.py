"""XDG locations and atomic JSON persistence.

Adapted from soundboard's `paths.py`, and for the same reason: every write goes
through `write_json`, which writes a temporary file in the same directory and
then `os.replace`s it into place. `os.replace` is atomic on POSIX, so a crash or
a full disk mid-write leaves the previous file intact rather than a truncated
one. Readers tolerate a missing or corrupt file by returning a default, so a bad
settings file degrades to "defaults" instead of a crash at startup.

Two locations here are load-bearing in a way soundboard's were not:

* `models_dir()` **must** contain the literal path component `models`. The ncnn
  binary refuses to start otherwise — it derives its internal `prepadding` from
  whether the `-m` argument contains the substring `models` or `models2`, and
  prints `unknown model dir type` and exits non-zero for anything else. This is
  not configurable, so the name is fixed here rather than in settings.
* `work_dir()` is under the *cache* root, not data. A run writes one PNG per
  tile there, which for a large image is hundreds of megabytes of pure scratch;
  it is deleted after every run and is safe to lose at any time.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

APP = "local-upscaler"


def _xdg(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return (Path(v) if v else default) / APP


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config")


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", Path.home() / ".local/share")


def cache_dir() -> Path:
    return _xdg("XDG_CACHE_HOME", Path.home() / ".cache")


def models_dir() -> Path:
    """Where downloaded `.param`/`.bin` pairs live.

    The trailing component must stay `models` — see the module docstring.
    """
    return data_dir() / "models"


def engine_dir() -> Path:
    """Where `--fetch-engine` puts a managed copy of the ncnn binary."""
    return data_dir() / "bin"


def work_dir() -> Path:
    """Scratch root for tile PNGs. Disposable."""
    return cache_dir() / "work"


def desktop_file() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(base) / "applications" / f"{APP}.desktop"


def settings_file() -> Path:
    return config_dir() / "settings.json"


def write_json(path: Path, payload: dict) -> None:
    """Atomically replace `path` with `payload`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)      # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default: dict | None = None) -> dict:
    """Read JSON, returning `default` for anything unreadable."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else (default or {})
    except (OSError, json.JSONDecodeError, ValueError):
        return default or {}


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """A free path like `<stem><suffix>`, `<stem>-2<suffix>`, ..."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate
