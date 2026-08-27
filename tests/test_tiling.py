"""Tests for the tile geometry.

Run:  python3 tests/test_tiling.py

The important test here is the round trip. `split` writes tiles, a stub "engine"
transforms them, and `reassemble` puts them back; the result must equal what the
same transform would have produced on the whole image, **bit for bit**.

Nearest-neighbour resizing is the right stub because it is exactly local: every
output pixel depends on exactly one input pixel, so tiling can never change the
answer for a correct implementation, and any off-by-one in a crop box, a paste
offset or a margin measured at the wrong scale changes it immediately. A real
model would smear such a bug into a faint seam that no automated test could
catch, which is exactly why the geometry is verified separately from the model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile  # noqa: E402

from PIL import Image  # noqa: E402

from local_upscaler.engine import tiling as tl  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def _noise(width, height, mode="RGB", seed=1):
    """A deterministic image with no smooth regions, so seams cannot hide."""
    import random
    rnd = random.Random(seed)
    n = len(mode)
    img = Image.new(mode, (width, height))
    img.putdata([tuple(rnd.randrange(256) for _ in range(n))
                 for _ in range(width * height)])
    return img


# ------------------------------------------------------------------ geometry
def test_cores_tile_exactly():
    print("\ncores tile the image exactly")
    for w, h, t, c in [(100, 100, 40, 8), (1920, 1080, 384, 24), (7, 3, 4, 2),
                       (512, 512, 512, 24), (1000, 10, 256, 24)]:
        tiles = tl.plan_tiles(w, h, t, c)
        covered = set()
        overlap = False
        for tile in tiles:
            x0, y0, x1, y1 = tile.core
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if (x, y) in covered:
                        overlap = True
                    covered.add((x, y))
        check(len(covered) == w * h and not overlap,
              f"{w}x{h} tile={t}: {len(tiles)} cores cover every pixel once")


def test_pad_contains_core_and_stays_in_bounds():
    print("\npadded regions are sane")
    w, h, ctx = 300, 200, 24
    ok_contains = ok_bounds = ok_margin = True
    for tile in tl.plan_tiles(w, h, 64, ctx):
        cx0, cy0, cx1, cy1 = tile.core
        px0, py0, px1, py1 = tile.pad
        ok_contains &= px0 <= cx0 and py0 <= cy0 and px1 >= cx1 and py1 >= cy1
        ok_bounds &= px0 >= 0 and py0 >= 0 and px1 <= w and py1 <= h
        # The margin is `ctx`, except where the image edge cuts it short.
        ok_margin &= (cx0 - px0) == min(ctx, cx0) and (px1 - cx1) == min(ctx, w - cx1)
    check(ok_contains, "every pad contains its core")
    check(ok_bounds, "every pad stays inside the image")
    check(ok_margin, "the margin is ctx px except where clamped by the edge")


def test_empty_and_degenerate():
    print("\ndegenerate inputs")
    check(tl.plan_tiles(0, 100, 64) == [], "zero width plans no tiles")
    check(tl.plan_tiles(100, 0, 64) == [], "zero height plans no tiles")
    check(len(tl.plan_tiles(10, 10, 999)) == 1, "a tile larger than the image gives one tile")
    check(len(tl.plan_tiles(10, 10, 64, 0)) == 1, "ctx=0 is allowed")
    t = tl.plan_tiles(10, 10, 64, 0)[0]
    check(t.core == t.pad == (0, 0, 10, 10), "with ctx=0 the pad equals the core")


def test_auto_tile_size():
    print("\nauto tile size")
    check(tl.auto_tile_size(64, 64) == tl.MIN_TILE, "tiny images clamp to MIN_TILE")
    check(tl.auto_tile_size(20000, 20000) == tl.MAX_TILE, "huge images clamp to MAX_TILE")
    t = tl.auto_tile_size(1920, 1080)
    check(tl.MIN_TILE <= t <= tl.MAX_TILE, f"1920x1080 -> {t}, inside the clamp")
    n = len(tl.plan_tiles(4000, 3000, tl.auto_tile_size(4000, 3000)))
    check(4 <= n <= 64, f"12 MP plans {n} tiles — enough for a moving bar, not absurd")
    check(all(tl.auto_tile_size(w, h) % tl.TILE_QUANTUM == 0
              for w, h in [(1920, 1080), (800, 600), (5000, 4000)]),
          "sizes are a multiple of TILE_QUANTUM")


# ------------------------------------------------------------- the round trip
def _round_trip(width, height, tile, ctx, scale, mode="RGB", seed=1):
    """split -> nearest-neighbour stub engine -> reassemble. Returns (got, want)."""
    src = _noise(width, height, mode, seed)
    tiles = tl.plan_tiles(width, height, tile, ctx)
    with tempfile.TemporaryDirectory(prefix="lu-tile-test-") as tmp:
        din, dout = Path(tmp) / "in", Path(tmp) / "out"
        tl.split(src, tiles, din)
        dout.mkdir()
        for t in tiles:                                   # the stub engine
            with Image.open(din / f"{t.name}.png") as im:
                w, h = im.size
                im.resize((w * scale, h * scale), Image.NEAREST).save(dout / f"{t.name}.png")
        got = tl.reassemble(tiles, dout, (width, height), scale, mode)
    want = src.resize((width * scale, height * scale), Image.NEAREST)
    return got, want


def test_round_trip_identity():
    print("\nround trip at scale 1 reproduces the source exactly")
    for w, h, t, c in [(200, 150, 64, 24), (1920, 1080, 384, 24), (17, 5, 8, 3),
                       (256, 256, 256, 24), (300, 300, 64, 0)]:
        got, want = _round_trip(w, h, t, c, 1)
        check(got.tobytes() == want.tobytes(),
              f"{w}x{h} tile={t} ctx={c} scale=1 is bit-exact")


def test_round_trip_scaled():
    print("\nround trip at scale > 1 matches a whole-image pass exactly")
    for w, h, t, c, s in [(200, 150, 64, 24, 4), (101, 97, 32, 12, 4),
                          (640, 360, 128, 24, 2), (77, 43, 16, 5, 3)]:
        got, want = _round_trip(w, h, t, c, s)
        check(got.size == want.size and got.tobytes() == want.tobytes(),
              f"{w}x{h} tile={t} ctx={c} scale={s} is bit-exact")


def test_round_trip_alpha():
    print("\nalpha survives the round trip")
    got, want = _round_trip(120, 90, 48, 16, 4, mode="RGBA")
    check(got.mode == "RGBA" and got.tobytes() == want.tobytes(),
          "RGBA 4x is bit-exact")


def test_wrong_scale_is_rejected():
    print("\na tile of the wrong size is a clear error, not a silent smear")
    src = _noise(64, 64)
    tiles = tl.plan_tiles(64, 64, 32, 8)
    with tempfile.TemporaryDirectory(prefix="lu-tile-test-") as tmp:
        din, dout = Path(tmp) / "in", Path(tmp) / "out"
        tl.split(src, tiles, din)
        dout.mkdir()
        for t in tiles:                     # engine produces 2x, caller says 4x
            with Image.open(din / f"{t.name}.png") as im:
                w, h = im.size
                im.resize((w * 2, h * 2), Image.NEAREST).save(dout / f"{t.name}.png")
        try:
            tl.reassemble(tiles, dout, (64, 64), 4)
            check(False, "reassemble raises on a mismatched tile size")
        except ValueError as e:
            check("expected" in str(e), f"reassemble raises ValueError: {e}")


def test_output_bytes():
    print("\noutput size estimate")
    check(tl.output_bytes(1000, 1000, 4) == 1000 * 4 * 1000 * 4 * 3,
          "1000x1000 at 4x is 48 MB of RGB")


def main():
    for fn in (test_cores_tile_exactly, test_pad_contains_core_and_stays_in_bounds,
               test_empty_and_degenerate, test_auto_tile_size,
               test_round_trip_identity, test_round_trip_scaled,
               test_round_trip_alpha, test_wrong_scale_is_rejected,
               test_output_bytes):
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
