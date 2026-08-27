# Compatibility

What this app needs, what it merely likes, and what it does not care about.
Written for CachyOS / Arch; the reasoning transfers to other distributions even
though the package names do not.

## Hard requirements

Without these the app does not start.

| Requirement | Package | Why |
|---|---|---|
| Python 3.11+ | `python` | `X \| Y` type syntax, `:=`, `dataclasses`. Developed on 3.14.7 |
| PySide6 | `pyside6` | The entire GUI. ABI-locked to `qt6-base` — never partial-upgrade |
| Pillow | `python-pillow` | Reads the source, cuts tiles, reassembles the result |
| An upscaling engine | `realesrgan-ncnn-vulkan` (AUR) or `--fetch-engine` | Does the actual work |

NumPy is **not** required at runtime. It is listed in the README's pacman line
only because Pillow is markedly better off with it present.

## Per-feature requirements

Missing these costs one feature, not the app.

| Requirement | Package | What you lose |
|---|---|---|
| A Vulkan ICD | `vulkan-icd-loader` plus a driver | GPU acceleration. **Advanced → Device: CPU only** still works, 10-50x slower |
| NVIDIA driver | `nvidia-utils` (or `nvidia-580xx-utils` for Pascal) | As above, on NVIDIA cards |
| `kbuildsycoca6` | `kservice` | The menu entry may not appear until the next login |
| `update-desktop-database` | `desktop-file-utils` | As above, on non-KDE desktops |
| `gtk-update-icon-cache` | `gtk-update-icon-cache` | Nothing — it is only used when it can be run correctly, and the cache is deleted otherwise |
| Network access | — | Downloading models and the engine. Once fetched, the app is fully offline |

## Hardware

| | Minimum | Notes |
|---|---|---|
| VRAM | ~1 GB | At the default GPU tile of 128. Peak measured 1.7 GB on a 1080p 4x run. Lower the tile to 64 for less |
| RAM | 4 GB, more for large images | A 12 MP source at 4x is a 192 MP result — about 580 MB as raw RGB, plus Qt's copy for display |
| Disk | ~500 MB | 11 MB engine, and 1-64 MB per model (431 MB for all sixteen) |
| GPU | Anything with Vulkan 1.1 | Including cards CUDA has dropped — see below |

### The Pascal / CUDA note

This app deliberately does **not** use PyTorch, and the reason is a hardware
constraint that is easy to trip over:

* CUDA 13.0 removed offline compilation for compute capability below 7.5 —
  Maxwell, Pascal and Volta.
* Arch's `python-pytorch-cuda` depends on `cuda` 13.x, so it ships no kernels for
  those cards. It will import, report a GPU, and run on the CPU.
* Upstream pip wheels are no better: Pascal was dropped from the cu128/cu129
  builds, and `cp314` wheels only exist for torch ≥ 2.10, which is CUDA 13.
* Driver branch 580 is the last one supporting Pascal at all. On such a machine,
  **do not** move off `nvidia-580xx-utils`.

ncnn goes through Vulkan, which none of that affects, so these cards remain fully
usable here.

## Does not matter

* **Shell.** Every subprocess is an argument list, never a shell string. Fish,
  bash, zsh — irrelevant.
* **Display server.** Wayland and X11 both work. Developed on KDE/Wayland.
* **Desktop environment.** KDE gets the nicest integration because `ui/metrics.py`
  reads `kdeglobals`, but it falls back to sensible values everywhere else.
* **GPU vendor.** Anything with a working Vulkan driver. Multi-GPU is not
  exposed in the UI, though the engine supports it.
* **Locale.** `QLocale` is pinned to English so dialog buttons do not come out
  half-translated on a `pt_BR` system.
* **Colour scheme.** Light and dark both work; icons are recoloured against the
  live palette and follow a theme change without a restart.

## Not supported

* **Windows and macOS.** Nothing here is deeply Linux-specific except the XDG
  paths and the `.desktop` installer, but neither is tested.
* **pip / virtualenv installs.** PySide6 from PyPI is not ABI-compatible with the
  system `qt6-base` that Plasma integration loads into the process, and mixing
  them produces crashes that look like Qt bugs. Use the distribution packages.
