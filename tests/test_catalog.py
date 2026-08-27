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
              and m.param_bytes > 0 and m.bin_bytes > 0
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
        check(m.download_bytes == m.param_bytes + m.bin_bytes,
              f"{m.id}: download_bytes is param + bin")
    total = sum(m.download_bytes for m in cat.MODELS)
    check(300e6 < total < 600e6, f"the whole catalog is {total/1048576:.0f} MB")


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
               test_download_size, test_filenames, test_only_animevideo_is_multiscale,
               test_base_urls, test_licences_are_stated):
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
