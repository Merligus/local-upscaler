"""Entry point.  python3 -m local_upscaler [image] [--options]

Bootstrap order matters and is not arbitrary (the same reasoning as soundboard's
entry point, which these lines are taken from):

* `setHighDpiScaleFactorRoundingPolicy` must precede `QApplication`.
* `setDesktopFileName` before any window — it is the Wayland app_id, and what
  the compositor matches against the installed `.desktop` file for the task
  switcher icon.
* `QLocale.setDefault(English)` is **not** optional here: the system locale is
  pt_BR, and without it QFileDialog and QMessageBox buttons come out in
  Portuguese while the rest of the UI is English.
* `app.setStyle()` is never called — plasma-integration already selects Breeze,
  the kdeglobals palette, the icon theme and the font.

Every heavy import is deferred inside the subcommand that needs it, so
`--help`, `--list-models` and `--fetch-*` never import PySide6.
"""

from __future__ import annotations

import sys
from pathlib import Path

USAGE = """Local Upscaler — enlarge images locally with AI upscaling models.

  python3 -m local_upscaler [IMAGE]     start the app, optionally on an image
  python3 -m local_upscaler --install   add it to the application menu
  python3 -m local_upscaler --uninstall remove the menu entry and icon
  python3 -m local_upscaler --list-models
                                        show the catalog and what is downloaded
  python3 -m local_upscaler --fetch-engine
                                        download the upscaling engine
  python3 -m local_upscaler --fetch-models MODEL...
                                        download models ('all' fetches everything)
  python3 -m local_upscaler --bench [MODEL...]
                                        measure this machine's actual speed
  python3 -m local_upscaler --help      this message
"""


#: Wall time of the last progress line, so a fast download does not emit one
#: per 256 kB chunk. Harmless on a terminal, where `\r` overwrites in place, but
#: it turns a redirected log into hundreds of lines of noise.
_last_progress = 0.0


def _progress_line(done: int, total: int, label: str) -> None:
    global _last_progress
    import time
    now = time.monotonic()
    finished = bool(total) and done >= total
    if not finished and now - _last_progress < 0.2:
        return
    _last_progress = now
    pct = f"{100 * done / total:5.1f}%" if total else "  ...."
    sys.stderr.write(f"\r  {label}: {done / 1048576:6.1f} / "
                     f"{total / 1048576:6.1f} MB  {pct}")
    if finished:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _list_models() -> int:
    from .engine import catalog, fetch

    print(f"{'MODEL':<32} {'SCALE':<8} {'SIZE':>8}  STATUS")
    for m in catalog.MODELS:
        scale = "/".join(f"{s}x" for s in m.scales)
        have = all(fetch.have_model(m, s) for s in m.scales)
        some = any(fetch.have_model(m, s) for s in m.scales)
        status = "downloaded" if have else ("partial" if some else "-")
        print(f"{m.id:<32} {scale:<8} {m.download_bytes / 1048576:7.1f}M  {status}")
        print(f"{'':<32} {m.blurb}")
    print()
    print("Fetch one with:  python3 -m local_upscaler --fetch-models <MODEL>")
    return 0


def _fetch_engine() -> int:
    from .engine import binary, fetch

    existing = binary.find()
    if existing is not None and existing != binary.managed_path():
        print(f"An engine is already installed at {existing}.")
        print("Fetching anyway; the system one keeps priority.")
    try:
        path = fetch.fetch_engine(progress=_progress_line)
    except fetch.FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if binary.probe(path) is None:
        print(f"error: {path} downloaded but does not run.", file=sys.stderr)
        return 1
    print(f"Engine installed at {path}")
    return 0


def _fetch_models(names: list[str]) -> int:
    from .engine import catalog, fetch

    if not names:
        print("error: name at least one model, or 'all'.", file=sys.stderr)
        return 2
    wanted = (list(catalog.MODELS) if "all" in names
              else [m for m in (catalog.by_id(n) for n in names) if m is not None])
    unknown = [] if "all" in names else [n for n in names if catalog.by_id(n) is None]
    for name in unknown:
        print(f"warning: no model called {name!r}", file=sys.stderr)
    if not wanted:
        return 1

    failed = 0
    for model in wanted:
        for scale in model.scales:
            if fetch.have_model(model, scale):
                continue
            try:
                fetch.fetch_model(model, scale, progress=_progress_line)
            except fetch.FetchError as e:
                print(f"\nerror: {model.id}: {e}", file=sys.stderr)
                failed += 1
        print(f"  {model.id}: ready")
    return 1 if failed else 0


def _bench(names: list[str]) -> int:
    """Measure real throughput, so the README's numbers are this machine's."""
    import tempfile
    import time

    from PIL import Image

    from . import settings as st
    from .engine import binary, catalog, fetch, runner

    if binary.find() is None:
        print("error: no engine found. Run --fetch-engine first.", file=sys.stderr)
        return 1
    models = ([catalog.get(n) for n in names] if names
              else [catalog.get("upscayl-standard-4x"), catalog.get("upscayl-lite-4x")])

    settings = st.load()
    sizes = ((512, 512), (1024, 1024))
    print(f"{'MODEL':<28} {'INPUT':>12} {'TILES':>6} {'WALL':>9} {'s/MP':>8}")
    with tempfile.TemporaryDirectory(prefix="lu-bench-") as tmp:
        for model in models:
            scale = model.default_scale()
            if not fetch.have_model(model, scale):
                print(f"  fetching {model.id}…", file=sys.stderr)
                try:
                    fetch.fetch_model(model, scale, progress=_progress_line)
                except fetch.FetchError as e:
                    print(f"error: {e}", file=sys.stderr)
                    continue
            for width, height in sizes:
                src = Path(tmp) / f"bench_{width}.png"
                if not src.exists():
                    # Structured content, not flat colour: a constant image would
                    # be unrepresentatively easy on memory bandwidth.
                    img = Image.new("RGB", (width, height))
                    img.putdata([((x * 7) % 256, (y * 5) % 256, (x ^ y) % 256)
                                 for y in range(height) for x in range(width)])
                    img.save(src)
                job = runner.Job(source=src, model=model, scale=scale,
                                 engine_tile=settings.engine_tile,
                                 outer_tile=runner.plan_tile_size(width, height),
                                 gpu=settings.gpu_arg())
                start = time.monotonic()
                try:
                    result = runner.Runner(job).run()
                except runner.UpscaleError as e:
                    print(f"{model.id:<28} {width}x{height}  FAILED: {e}")
                    continue
                elapsed = time.monotonic() - start
                print(f"{model.id:<28} {width}x{height:>7} {result.tiles:>6} "
                      f"{elapsed:>8.1f}s {result.sec_per_mpx:>7.1f}")
                settings.calibration.record(model.id, scale, result.sec_per_mpx)
    st.save(settings)
    print("\nRecorded. The app's time estimates now use these numbers.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    args = argv[1:]

    if "--help" in args or "-h" in args:
        print(USAGE)
        return 0
    if "--install" in args:
        from .install import install
        return install()
    if "--uninstall" in args:
        from .install import uninstall
        return uninstall()
    if "--list-models" in args:
        return _list_models()
    if "--fetch-engine" in args:
        return _fetch_engine()
    if "--fetch-models" in args:
        return _fetch_models(args[args.index("--fetch-models") + 1:])
    if "--bench" in args:
        return _bench(args[args.index("--bench") + 1:])

    unknown = [a for a in args if a.startswith("-")]
    if unknown:
        print(f"error: unknown option {unknown[0]}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    from PySide6.QtCore import QLocale, Qt
    from PySide6.QtGui import QGuiApplication, QIcon
    from PySide6.QtWidgets import QApplication

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(argv)
    app.setApplicationName("Local Upscaler")
    app.setApplicationDisplayName("Local Upscaler")
    app.setOrganizationName("local-upscaler")
    app.setDesktopFileName("local-upscaler")
    QLocale.setDefault(QLocale(QLocale.Language.English,
                               QLocale.Country.UnitedStates))
    QIcon.setFallbackThemeName("breeze")

    from .ui import icons, metrics
    icons.install(app)
    metrics.install(app)

    app_icon = QIcon.fromTheme("local-upscaler")
    if app_icon.isNull():
        app_icon = icons.icon("local-upscaler", 48)
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    from .ui.main_window import MainWindow
    window = MainWindow()
    positional = [a for a in args if not a.startswith("-")]
    if positional:
        window.open_path(Path(positional[0]).expanduser())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
