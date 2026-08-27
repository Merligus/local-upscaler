"""Splitting an image into overlapping tiles, and putting it back together.

This module exists for one reason that is not obvious: **the progress bar**.

`realesrgan-ncnn-vulkan` prints nothing at all while it works. With `-v` it
prints exactly one line per *file* it finishes, from its save thread:

    <in> -> <out> done

So there is no way to watch a single large image make progress. The way out is
to hand the binary many files instead of one: cut the source into tiles, drop
them in a directory, run the binary **once** in directory mode, and count the
`done` lines. That gives a genuinely determinate bar, a cancel that takes effect
promptly, and a memory ceiling that does not depend on the input size — for one
subprocess launch and one model load, not one per tile.

The correctness risk this introduces is seams. Each tile is therefore cut *with*
a margin of surrounding context, upscaled, and then the margin is cropped back
off before the tile's own core is pasted into the canvas. Only the core is kept,
so neighbouring tiles agree by construction on who owns which output pixel;
the margin exists purely to give the convolutions the same neighbourhood they
would have seen in a whole-image pass.

`DEFAULT_CONTEXT = 48` is **measured, not guessed**. Upstream's internal tiling
uses a `prepadding` of only 10 px, which suggested a much smaller margin would
do. It does not.

The measurement: a 256x256 crop of a photograph, upscaled 4x with
`upscayl-standard-4x`, tiled here at 128 px so boundaries fall every 512 output
pixels, and compared against the same image upscaled with the engine's tile set
to the full width so that it does no tiling at all. "band" is the mean absolute
error in an 8 px strip on the tile boundaries, "else" the same everywhere else.

    ctx    max abs diff    band/else
      0        70            5.05x     a plainly visible seam
     24         4            3.10x     measurable
     48         1            1.32x     <- shipped
     96         1            1.03x     no better where it counts

The number that decides it is the maximum, not the ratio. At 48 the worst pixel
in the image differs by a single quantisation level, which cannot be displayed,
let alone seen; the residual 1.32x ratio is the quotient of two quantities that
are both far below perceptibility. Going to 96 flattens that ratio but does not
improve the maximum, and it is not free: the margin is added on both sides of
every tile, so at a 384 px tile it takes the processed area from 1.56x the useful
area to 2.25x. That is a 44% longer run for a change from invisible to
invisible.

One trap worth recording, because it cost an hour and would cost it again.
Comparing against a reference that was itself produced with the binary's
internal tiling (`-t 128`) appeared to show a 3x seam at ctx=24. It was an
artifact: at tile 384 the tile boundaries land every 1536 output px, and the
reference's own internal seams land every 512 output px, so the two coincided
and the measurement was reading the *reference's* seams. Any future check of
this kind must use an untiled reference, or at minimum an internal tile size
that is not a divisor of the outer one.

`plan_tiles` is pure arithmetic and `split`/`reassemble` are exact inverses of
one another at scale 1. `tests/test_tiling.py` asserts that round trip
bit-exactly against a stub engine, because every plausible bug in this file — an
off-by-one in a crop, a margin cropped at source scale instead of output scale —
shows up there as a mismatched pixel rather than as a subtle smear that only a
human eye catches weeks later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

#: Context margin in source pixels, added around every tile and cropped back off.
#: See the module docstring — this value is measured, and 24 is not enough.
DEFAULT_CONTEXT = 48

#: How many tiles `auto_tile_size` aims for. Purely a progress-granularity knob:
#: the work is the same, this only decides how often the bar moves.
TARGET_TILES = 16

#: Tiles are clamped to this range. The lower bound stops small images from
#: paying a large context overhead; the upper bound keeps the bar moving on
#: very large ones.
MIN_TILE = 384
MAX_TILE = 1024

#: Tile dimensions are rounded to a multiple of this, purely for tidiness.
TILE_QUANTUM = 64


@dataclass(frozen=True)
class Tile:
    """One tile: the region it owns, and the larger region actually processed.

    `core` is the half-open source rectangle this tile is responsible for.
    `pad` is `core` grown by the context margin and clamped to the image, and is
    what gets written out and sent to the engine.
    """

    index: int
    core: tuple[int, int, int, int]
    pad: tuple[int, int, int, int]

    @property
    def name(self) -> str:
        """Filename stem. Zero-padded so directory order matches tile order."""
        return f"{self.index:05d}"

    @property
    def pad_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.pad
        return x1 - x0, y1 - y0

    @property
    def core_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.core
        return x1 - x0, y1 - y0

    def crop_box(self, scale: int) -> tuple[int, int, int, int]:
        """Where `core` sits inside this tile's upscaled output."""
        cx0, cy0, cx1, cy1 = self.core
        px0, py0, _, _ = self.pad
        left = (cx0 - px0) * scale
        top = (cy0 - py0) * scale
        return left, top, left + (cx1 - cx0) * scale, top + (cy1 - cy0) * scale

    def paste_at(self, scale: int) -> tuple[int, int]:
        """Where `core` belongs in the output canvas."""
        cx0, cy0, _, _ = self.core
        return cx0 * scale, cy0 * scale


def auto_tile_size(width: int, height: int, target: int = TARGET_TILES) -> int:
    """A tile size that cuts a `width` x `height` image into roughly `target` tiles.

    The context margin is a fixed number of pixels, so it costs proportionally
    more on small tiles. That works out well rather than badly: this returns
    small tiles only for small images, where the run is short and the overhead is
    a few seconds, and large tiles for large images, where the overhead would
    actually matter but is now only a few percent.
    """
    if width <= 0 or height <= 0:
        return MIN_TILE
    ideal = math.sqrt((width * height) / max(1, target))
    quantised = round(ideal / TILE_QUANTUM) * TILE_QUANTUM
    return max(MIN_TILE, min(MAX_TILE, int(quantised)))


def plan_tiles(width: int, height: int, tile: int,
               ctx: int = DEFAULT_CONTEXT) -> list[Tile]:
    """Cut `width` x `height` into tiles of at most `tile` px plus `ctx` context.

    Cores tile the image exactly — they do not overlap and they leave no gap —
    so the pasted result covers every output pixel exactly once.
    """
    if width <= 0 or height <= 0:
        return []
    tile = max(1, int(tile))
    ctx = max(0, int(ctx))

    tiles: list[Tile] = []
    index = 0
    for y0 in range(0, height, tile):
        y1 = min(y0 + tile, height)
        for x0 in range(0, width, tile):
            x1 = min(x0 + tile, width)
            pad = (max(0, x0 - ctx), max(0, y0 - ctx),
                   min(width, x1 + ctx), min(height, y1 + ctx))
            tiles.append(Tile(index=index, core=(x0, y0, x1, y1), pad=pad))
            index += 1
    return tiles


def split(image: Image.Image, tiles: list[Tile], out_dir: Path) -> None:
    """Write each tile's padded region to `out_dir` as a PNG.

    PNG because it is lossless: a JPEG round trip here would feed the model
    compression artifacts that are not in the source.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tiles:
        image.crop(t.pad).save(out_dir / f"{t.name}.png", "PNG", compress_level=1)


def reassemble(tiles: list[Tile], in_dir: Path, size: tuple[int, int],
               scale: int, mode: str = "RGB") -> Image.Image:
    """Crop the context off each upscaled tile and paste its core into a canvas.

    `size` is the source size; the canvas is `size * scale`.
    """
    width, height = size
    canvas = Image.new(mode, (width * scale, height * scale))
    for t in tiles:
        path = in_dir / f"{t.name}.png"
        with Image.open(path) as raw:
            piece = raw.convert(mode) if raw.mode != mode else raw.copy()
        expect = (t.pad_size[0] * scale, t.pad_size[1] * scale)
        if piece.size != expect:
            raise ValueError(
                f"tile {t.name}: engine returned {piece.size}, expected {expect}. "
                f"The model's actual scale probably is not {scale}x.")
        canvas.paste(piece.crop(t.crop_box(scale)), t.paste_at(scale))
    return canvas


def output_bytes(width: int, height: int, scale: int, channels: int = 3) -> int:
    """Uncompressed size of the result. Used to warn before a run starts."""
    return width * scale * height * scale * channels
