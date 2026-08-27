"""Tests for which image formats the app accepts.

Run:  python3 tests/test_image_formats.py

This file exists because of a reported bug: the app refused `.jfif` files. There
was no technical reason — JFIF is not a separate format, it is the container
ordinary JPEG has always used, and both Pillow and Qt read those files without
complaint. Windows and Edge hand out `.jfif` when you save an image from the
web, so it is a common thing to have. The only thing rejecting them was a
hand-written list of extensions in `ui/setup_page.py` that happened to omit one
alias of a format the app fully supported.

The guard against that recurring is `test_nothing_readable_is_hidden`, which
asks the two libraries what they can open and requires the dialog to offer all
of it. A test that merely checked for `.jfif` would only pin the one alias
someone already noticed.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(tempfile.mkdtemp(prefix="lu-formats-test-"))
os.environ["XDG_CONFIG_HOME"] = str(_ROOT / "config")
os.environ["XDG_DATA_HOME"] = str(_ROOT / "data")
os.environ["XDG_CACHE_HOME"] = str(_ROOT / "cache")

import mimetypes  # noqa: E402

from PIL import Image  # noqa: E402
from PySide6.QtGui import QImageReader  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

FAILS = []
APP = QApplication.instance() or QApplication(sys.argv)

from local_upscaler import settings as st  # noqa: E402
from local_upscaler.ui import icons as ic, metrics as me  # noqa: E402

ic.install(APP)
me.install(APP)

from local_upscaler.ui.setup_page import (SetupPage, image_filter,  # noqa: E402
                                          readable_extensions)

SAMPLES = _ROOT / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def _sample(width=64, height=48):
    """A small image with actual content, so encoders do not shortcut."""
    img = Image.new("RGB", (width, height))
    img.putdata([((x * 7) % 256, (y * 11) % 256, (x ^ y) % 256)
                 for y in range(height) for x in range(width)])
    return img


def test_jfif_is_offered():
    """The reported bug, pinned directly."""
    print("\n.jfif is offered (the reported bug)")
    every = readable_extensions()
    check("jfif" in every, "'jfif' is in the readable set")
    check("*.jfif" in image_filter(), "'*.jfif' appears in the dialog filter")

    # It is JPEG. If jpg is offered, every alias of it must be too.
    for alias in ("jpg", "jpeg", "jfif", "jpe"):
        check(alias in every, f"JPEG alias '{alias}' is offered")


def test_filter_shape():
    print("\nthe filter is well formed")
    f = image_filter()
    check(f.startswith("Images (*."), "it starts with a named Images group")
    check(f.endswith(";;All files (*)"),
          "it ends with an All files escape hatch, so nothing is unreachable")
    check(f.count(";;") == 1, "there are exactly two filter groups")
    # Common formats are listed first so the dialog is readable.
    patterns = f.split(";;")[0]
    check(patterns.index("*.png") < patterns.index("*.xbm")
          if "*.xbm" in patterns else True,
          "common formats come before obscure ones")


def test_nothing_readable_is_hidden():
    """The real guard: the dialog must offer everything both libraries can open.

    Derived from the libraries rather than from a list, so a format alias cannot
    be forgotten the way `.jfif` was.
    """
    print("\nno format both libraries can read is hidden from the dialog")
    Image.init()
    qt_mimes = {bytes(m).decode().lower() for m in QImageReader.supportedMimeTypes()}
    every = set(readable_extensions())

    missing = []
    for extension, plugin in Image.EXTENSION.items():
        if plugin not in Image.OPEN:
            continue
        mime, _ = mimetypes.guess_type("x" + extension)
        if mime and mime in qt_mimes and extension.lstrip(".").lower() not in every:
            missing.append(extension)
    check(not missing, f"nothing readable is omitted (missing: {sorted(missing)})")
    check(len(every) > 10, f"the set is populated ({len(every)} extensions)")


def test_offered_formats_actually_load():
    """Every offered extension must survive the real selection path."""
    print("\nevery offered format is accepted by load_image")
    page = SetupPage(st.Settings())
    base = _sample()
    checked = skipped = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for extension in readable_extensions():
            path = SAMPLES / f"sample.{extension}"
            source = base
            if extension in ("pbm", "xbm"):
                source = base.convert("1")
            elif extension == "pgm":
                source = base.convert("L")
            try:
                source.save(path)
            except Exception:                       # noqa: BLE001
                # Pillow reads some formats it cannot write (psd, xpm). Not an
                # app limitation, so there is nothing to assert here.
                skipped += 1
                continue
            checked += 1
            if not page.load_image(path):
                check(False, f".{extension}: load_image rejected a valid file "
                             f"({page._dims.text()})")
    check(True, f"{checked} formats accepted, {skipped} unwritable by Pillow")


def test_jfif_round_trips():
    print("\na real .jfif behaves exactly like a .jpg")
    page = SetupPage(st.Settings())
    img = _sample(120, 90)
    as_jpg = SAMPLES / "same.jpg"
    as_jfif = SAMPLES / "same.jfif"
    img.save(as_jpg, "JPEG", quality=92)
    as_jfif.write_bytes(as_jpg.read_bytes())        # byte-identical, renamed

    check(page.load_image(as_jpg), ".jpg is accepted")
    jpg_size = page.source_size
    check(page.load_image(as_jfif), ".jfif is accepted")
    check(page.source_size == jpg_size,
          f"both report the same dimensions {page.source_size}")
    check(page.source == as_jfif, "the .jfif is the selected source")

    # Pillow must agree, because it does the tiling.
    with Image.open(as_jfif) as opened:
        check(opened.format == "JPEG", "Pillow identifies it as JPEG by content")


def test_bad_files_are_rejected_with_a_reason():
    print("\na file that is not an image is refused, and says why")
    page = SetupPage(st.Settings())
    junk = SAMPLES / "notanimage.jfif"
    junk.write_bytes(b"<html>404 not found</html>")
    check(not page.load_image(junk), "an HTML page named .jfif is rejected")
    check(page.source is None, "no source is left selected")
    # The reason has to survive the refresh that follows it — it did not
    # originally, and the user saw only the generic "choose an image" line.
    check("notanimage.jfif" in page._dims.text(),
          f"the message names the file: {page._dims.text()!r}")
    check("Could not read" in page._name.text(), "the heading reports the failure")

    empty = SAMPLES / "empty.png"
    empty.write_bytes(b"")
    check(not page.load_image(empty), "an empty file is rejected")


def test_desktop_entry_covers_jpeg():
    print("\nOpen With covers .jfif through the system mime database")
    from local_upscaler import install as inst
    # .jfif maps to image/jpeg in shared-mime-info, so declaring image/jpeg is
    # what makes the file manager offer this app for those files. The mime list
    # is assembled from MIME_TYPES rather than written into the template, since
    # a .desktop MimeType line cannot be wrapped.
    check("image/jpeg" in inst.MIME_TYPES, "the desktop entry declares image/jpeg")
    check("{mime_types}" in inst.DESKTOP_TEMPLATE,
          "the template takes its mime list from MIME_TYPES")
    guessed, _ = mimetypes.guess_type("x.jfif")
    check(guessed == "image/jpeg", f".jfif resolves to {guessed}")

    # The rendered line must be a single line, or the entry is malformed.
    rendered = inst.DESKTOP_TEMPLATE.format(
        exec_line="/x", try_exec="/x", icon="i",
        mime_types=";".join(inst.MIME_TYPES) + ";")
    mime_line = next(ln for ln in rendered.splitlines() if ln.startswith("MimeType="))
    check(mime_line.endswith(";"), "the MimeType line is semicolon-terminated")
    check(all(m in mime_line for m in inst.MIME_TYPES),
          f"all {len(inst.MIME_TYPES)} types reach the rendered entry")


def main():
    for fn in (test_jfif_is_offered, test_filter_shape, test_nothing_readable_is_hidden,
               test_offered_formats_actually_load, test_jfif_round_trips,
               test_bad_files_are_rejected_with_a_reason, test_desktop_entry_covers_jpeg):
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
