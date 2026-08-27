"""Tests for the model catalog.

Run:  python3 tests/test_catalog.py

Pure data, so this runs with no network and no display. What it guards is the
class of mistake that only shows up as a failed download or an ncnn error much
later: a duplicated id, a scale a model cannot produce, a URL pointing at the
wrong repository, or the `realesr-animevideov3` filename special case being
quietly broken by an edit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_upscaler.engine import catalog as cat  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def test_identity():
    print("\nidentity")
    ids = [m.id for m in cat.MODELS]
    check(len(ids) == len(set(ids)), "model ids are unique")
    labels = [m.label for m in cat.MODELS]
    check(len(labels) == len(set(labels)), "labels are unique")
    check(cat.by_id(cat.DEFAULT_MODEL_ID) is not None,
          f"the default model {cat.DEFAULT_MODEL_ID!r} exists")
    check(cat.get("no-such-model").id == cat.DEFAULT_MODEL_ID,
          "an unknown id falls back to the default instead of raising")
    check(cat.by_id("no-such-model") is None, "by_id returns None for an unknown id")


def test_fields_are_populated():
    print("\nevery model is fully described")
    for m in cat.MODELS:
        ok = (m.label and m.blurb and m.author and m.licence
              and m.url.startswith("https://")
              and m.base_url.startswith("https://")
              and m.sizes and all(p > 0 and b > 0 for p, b in m.sizes.values())
              and m.sec_per_mpx > 0 and m.startup_s > 0
              and m.scales and all(1 <= s <= 8 for s in m.scales))
        check(bool(ok), f"{m.id} has every field populated and sane")


def test_blurbs_are_useful():
    print("\nblurbs say what the model is for")
    for m in cat.MODELS:
        check(20 <= len(m.blurb) <= 100 and m.blurb.endswith("."),
              f"{m.id}: blurb is a usable one-liner ({len(m.blurb)} chars)")


def test_download_size():
    print("\ndownload sizes")
    for m in cat.MODELS:
        param, binary = m.file_sizes(m.default_scale())
        check(m.download_bytes() == param + binary,
              f"{m.id}: download_bytes is param + bin")
    total = sum(sum(m.download_bytes(s) for s in m.scales) for m in cat.MODELS)
    check(300e6 < total < 600e6, f"the whole catalog is {total/1048576:.0f} MB")


def test_sizes_are_declared_for_every_scale():
    """The regression guard for the animevideov3 x4 download failure.

    `sizes` used to be a single (param, bin) pair per model. That is wrong for
    any model whose scale variants are different files: realesr-animevideov3's
    x4 parameter file is 3077 bytes where its x2 and x3 are 3173, so every x4
    download failed its size check with "expected 3173 bytes, got 3077" — and
    x4 is the default scale, so the model was simply unusable.
    """
    print("\nsizes are declared per scale, not per model")
    for m in cat.MODELS:
        check(set(m.sizes) == set(m.scales),
              f"{m.id}: declares a size for exactly its supported scales "
              f"({sorted(m.sizes)} vs {list(m.scales)})")
        for scale in m.scales:
            param, binary = m.file_sizes(scale)
            check(param > 0 and binary > 0,
                  f"{m.id} @{scale}x: both files have a positive size")

    # The specific case, pinned to the values read from the server.
    anime = cat.get(cat.SCALED_FILENAME_MODEL)
    check(anime.file_sizes(2) == (3173, 1247368), "animevideov3 x2 is (3173, 1247368)")
    check(anime.file_sizes(3) == (3173, 1247368), "animevideov3 x3 is (3173, 1247368)")
    check(anime.file_sizes(4) == (3077, 1247368),
          "animevideov3 x4 is (3077, 1247368) — different from x2/x3, which is the bug")
    check(anime.file_sizes(2) != anime.file_sizes(4),
          "the scales genuinely differ, so one pair per model cannot be right")

    # An unknown scale must fall back rather than raise, so a stale settings
    # file cannot crash the model picker.
    check(anime.file_sizes(9) == anime.file_sizes(anime.default_scale()),
          "an unsupported scale falls back to the default instead of raising")


def test_filenames():
    print("\nfilename derivation")
    m = cat.get("upscayl-standard-4x")
    check(m.filenames(4) == ("upscayl-standard-4x.param", "upscayl-standard-4x.bin"),
          "an ordinary model maps to <id>.param / <id>.bin")

    # The binary hardcodes a different path for exactly this one model name.
    a = cat.get(cat.SCALED_FILENAME_MODEL)
    check(a is not None and a.id == cat.SCALED_FILENAME_MODEL,
          f"{cat.SCALED_FILENAME_MODEL} is in the catalog")
    for scale in (2, 3, 4):
        want = (f"realesr-animevideov3-x{scale}.param",
                f"realesr-animevideov3-x{scale}.bin")
        check(a.filenames(scale) == want, f"animevideov3 at {scale}x -> {want[0]}")
    check(a.scales == (2, 3, 4), "animevideov3 is the model offering 2x and 3x")
    check(a.stem(3) == "realesr-animevideov3-x3", "stem carries the scale suffix")


def test_only_animevideo_is_multiscale():
    print("\nscale support")
    multi = [m.id for m in cat.MODELS if len(m.scales) > 1]
    check(multi == [cat.SCALED_FILENAME_MODEL],
          f"only {cat.SCALED_FILENAME_MODEL} offers more than one scale, got {multi}")
    for m in cat.MODELS:
        check(m.default_scale() in m.scales, f"{m.id}: default_scale is supported")


def test_base_urls():
    print("\nsource repositories")
    for m in cat.MODELS:
        ok = m.base_url in (cat._UPSCAYL, cat._CUSTOM)
        check(ok, f"{m.id} comes from a known repository")
    check(any(m.base_url == cat._UPSCAYL for m in cat.MODELS)
          and any(m.base_url == cat._CUSTOM for m in cat.MODELS),
          "both repositories are represented")


def test_licences_are_stated():
    print("\nlicensing is visible")
    for m in cat.MODELS:
        check(bool(m.licence.strip()), f"{m.id} states a licence")
    nc = [m.id for m in cat.MODELS if "NC" in m.licence]
    check(len(nc) > 0,
          f"non-commercial models are labelled as such ({len(nc)} of them)")


def main():
    for fn in (test_identity, test_fields_are_populated, test_blurbs_are_useful,
               test_download_size, test_sizes_are_declared_for_every_scale,
               test_filenames, test_only_animevideo_is_multiscale,
               test_base_urls, test_licences_are_stated):
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
