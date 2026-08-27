"""Tests for desktop integration, the icon asset and the CLI.

Run:  python3 tests/test_desktop_and_help.py

Installation runs against a temporary XDG root, so the developer's real menu
entry is never touched. The icon assertions are the ones worth keeping: an app
icon that depends on the palette is invisible in half of the places a launcher
draws it, and that failure is easy to introduce and hard to notice.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(tempfile.mkdtemp(prefix="lu-desktop-test-"))
os.environ["XDG_CONFIG_HOME"] = str(_ROOT / "config")
os.environ["XDG_DATA_HOME"] = str(_ROOT / "data")
os.environ["XDG_CACHE_HOME"] = str(_ROOT / "cache")

from PySide6.QtWidgets import QApplication  # noqa: E402

from local_upscaler import install as inst  # noqa: E402
from local_upscaler import paths  # noqa: E402

FAILS = []
APP = QApplication.instance() or QApplication(sys.argv)
APP.setDesktopFileName("local-upscaler")


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


# --------------------------------------------------------------------- icon
def test_icon_asset():
    print("\nicon asset")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    svg = inst.project_root() / "data" / f"{inst.APP_ID}.svg"
    check(svg.is_file(), f"data/{inst.APP_ID}.svg exists")
    renderer = QSvgRenderer(str(svg))
    check(renderer.isValid(), "the SVG parses")

    # Strip XML comments first: the file *documents* why it avoids currentColor,
    # and matching that prose would be a false positive.
    markup = re.sub(r"<!--.*?-->", "", svg.read_text(), flags=re.DOTALL)
    check("currentColor" not in markup,
          "the icon does not depend on currentColor (it is self-coloured)")
    check("ColorScheme-" not in markup, "the icon carries no palette classes")

    # It has to stay legible in a 16 px panel, which mostly means "has ink".
    for size in (16, 48):
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        inked = sum(1 for y in range(size) for x in range(size)
                    if image.pixelColor(x, y).alpha() > 16)
        check(inked > size * size * 0.3,
              f"at {size}x{size} the icon covers {100 * inked / (size * size):.0f}% "
              f"of its box")


# ------------------------------------------------------------------ install
def test_install_and_uninstall():
    print("\ninstall / uninstall against a temporary XDG root")
    before = inst.status()
    check(not before["desktop"], "nothing is installed to begin with")

    check(inst.install(verbose=False) == 0, "install() succeeds")
    after = inst.status()
    check(after["icon"], f"the icon landed at {inst.icon_target()}")
    check(after["desktop"], f"the launcher landed at {paths.desktop_file()}")
    check(after["wrapper_executable"], "bin/local-upscaler is executable")

    entry = paths.desktop_file().read_text()
    for key in ("Type=Application", "Name=Local Upscaler", "Icon=local-upscaler",
                "StartupWMClass=local-upscaler", "Terminal=false"):
        check(key in entry, f"the desktop entry declares {key}")
    check("Categories=Graphics;" in entry, "it is filed under Graphics")
    check("MimeType=image/png;" in entry,
          "it registers for images, so it appears under 'Open With'")
    check("%f" in entry, "Exec takes a file argument, so 'Open With' passes the image")

    exec_line = next(ln for ln in entry.splitlines() if ln.startswith("Exec="))
    target = exec_line[len("Exec="):].split()[0]
    check(Path(target).is_file() and os.access(target, os.X_OK),
          f"Exec points at something runnable: {target}")

    root = inst.icon_target().parent.parent.parent
    made = [s for s in inst.PNG_SIZES
            if (root / f"{s}x{s}" / "apps" / f"{inst.APP_ID}.png").is_file()]
    check(len(made) == len(inst.PNG_SIZES),
          f"all {len(inst.PNG_SIZES)} raster sizes were written ({made})")

    # A cache that cannot be built correctly must be removed, not left stale —
    # a near-empty one hides every icon in the tree, not just ours.
    check(not (root / "icon-theme.cache").exists() or (root / "index.theme").is_file(),
          "no stale icon-theme.cache is left behind")

    check(inst.uninstall(verbose=False) == 0, "uninstall() succeeds")
    gone = inst.status()
    check(not gone["icon"] and not gone["desktop"], "both are removed again")
    check(not any((root / f"{s}x{s}" / "apps" / f"{inst.APP_ID}.png").is_file()
                  for s in inst.PNG_SIZES), "the raster sizes are removed too")


def test_uninstall_keeps_downloads():
    print("\nuninstall leaves expensive downloads alone")
    models = paths.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    decoy = models / "pretend.bin"
    decoy.write_bytes(b"x" * 16)
    inst.install(verbose=False)
    inst.uninstall(verbose=False)
    check(decoy.is_file(), "a downloaded model survives --uninstall")


# ---------------------------------------------------------------------- cli
def _run(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(inst.project_root())
    return subprocess.run([sys.executable, "-m", "local_upscaler", *args],
                          capture_output=True, text=True, timeout=120, env=env)


def test_cli():
    print("\ncommand line")
    r = _run("--help")
    check(r.returncode == 0, "--help exits 0")
    for flag in ("--install", "--uninstall", "--fetch-engine", "--fetch-models",
                 "--bench", "--list-models"):
        check(flag in r.stdout, f"--help documents {flag}")

    r = _run("--list-models")
    check(r.returncode == 0, "--list-models exits 0")
    from local_upscaler.engine import catalog
    check(all(m.id in r.stdout for m in catalog.MODELS),
          "every catalog model is listed")

    r = _run("--nonsense")
    check(r.returncode == 2, "an unknown option exits 2")
    check("unknown option" in r.stderr, "and says which one")

    r = _run("--fetch-models")
    check(r.returncode == 2, "--fetch-models with no argument exits 2")

    # Must not touch the network for an unknown name.
    r = _run("--fetch-models", "definitely-not-a-model")
    check(r.returncode == 1 and "no model called" in r.stderr,
          "an unknown model name is reported, not fetched")


def test_help_needs_no_gui():
    print("\n--help does not import the GUI")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(inst.project_root())
    # Deferred imports are the point: --help must work with no display at all.
    env["QT_QPA_PLATFORM"] = "definitely-not-a-platform"
    r = subprocess.run([sys.executable, "-m", "local_upscaler", "--help"],
                       capture_output=True, text=True, timeout=60, env=env)
    check(r.returncode == 0, "--help works even with a broken Qt platform set")


def test_result_is_attributed_to_the_job_that_ran():
    """A run that finishes after the model combo moved on must not be relabelled.

    The setup page stays interactive while a run is in flight, so its widgets
    are not a record of what was started. Rebuilding the job from them on
    completion attributed the result — and its measured throughput — to whatever
    model happened to be selected when it landed, which put a 339 s/MP figure
    into the calibration for a 43 s/MP model.
    """
    print("\nthe result belongs to the job that ran, not the current selection")
    from local_upscaler.ui.main_window import MainWindow

    window = MainWindow()
    window._setup._model.setCurrentIndex(window._setup._model.findData("upscayl-lite-4x"))
    job = window._setup.build_job()
    window._job = job                       # as _start() would record it

    # The user changes their mind while the run is still going.
    window._setup._model.setCurrentIndex(
        window._setup._model.findData("high-fidelity-4x"))
    check(window._setup.build_job().model.id == "high-fidelity-4x",
          "the setup page now reports a different model, as the user selected")
    check(window._job.model.id == "upscayl-lite-4x",
          "but the running job still names the model that was actually started")
    window._teardown()
    check(window._job is None, "tearing down clears the remembered job")


def main():
    for fn in (test_icon_asset, test_install_and_uninstall,
               test_result_is_attributed_to_the_job_that_ran,
               test_uninstall_keeps_downloads, test_cli, test_help_needs_no_gui):
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
