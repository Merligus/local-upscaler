"""Tests for settings persistence and the ETA calibration.

Run:  python3 tests/test_settings.py

Runs against a temporary XDG root, so the developer's real settings are never
touched. The theme is tolerance: a settings file is a plain JSON file a user can
edit, a disk can truncate and a future version can extend, and none of those
should be able to stop the app from starting.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(tempfile.mkdtemp(prefix="lu-settings-test-"))
os.environ["XDG_CONFIG_HOME"] = str(_ROOT / "config")
os.environ["XDG_DATA_HOME"] = str(_ROOT / "data")
os.environ["XDG_CACHE_HOME"] = str(_ROOT / "cache")

from local_upscaler import paths, settings as st  # noqa: E402
from local_upscaler.engine import catalog, runner  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def test_defaults():
    print("\ndefaults")
    s = st.Settings()
    check(s.model_id == catalog.DEFAULT_MODEL_ID,
          "the default model is the catalog default")
    check(s.engine_tile == st.DEFAULT_ENGINE_TILE, "engine tile defaults to 128")
    check(s.outer_tile is None, "outer tile defaults to auto")
    check(s.device == "auto" and s.gpu_arg() is None,
          "device defaults to auto (-g omitted)")
    check(st.Settings(device="cpu").gpu_arg() == -1, "device=cpu maps to -g -1")
    check(s.compare_filter == "nearest", "the comparison shows real pixels by default")


def test_round_trip():
    print("\nround trip")
    s = st.Settings(model_id="ultrasharp-4x", scale=4, engine_tile=192,
                    outer_tile=512, tta=True, device="cpu",
                    compare_filter="smooth", last_save_dir="/tmp/x")
    s.calibration.record("ultrasharp-4x", 4, 49.4)
    back = st.Settings.from_dict(s.to_dict())
    same = (back.model_id == s.model_id and back.engine_tile == 192
            and back.outer_tile == 512 and back.tta is True
            and back.device == "cpu" and back.compare_filter == "smooth"
            and back.last_save_dir == "/tmp/x")
    check(same, "every field survives a to_dict/from_dict round trip")
    check(abs((back.calibration.get("ultrasharp-4x", 4) or 0) - 49.4) < 0.01,
          "calibration survives the round trip")


def test_garbage_tolerance():
    print("\ngarbage on disk degrades to defaults")
    s = st.Settings.from_dict({"model_id": "deleted-in-v2", "scale": "banana",
                               "engine_tile": None, "device": "quantum",
                               "compare_filter": 7, "outer_tile": "x",
                               "calibration": "not a dict"})
    check(s.model_id == catalog.DEFAULT_MODEL_ID,
          "an unknown model falls back to the default")
    check(s.scale in s.model().scales, "a nonsense scale falls back to a supported one")
    check(s.engine_tile == st.DEFAULT_ENGINE_TILE, "a null engine tile falls back")
    check(s.device == "auto", "an unknown device falls back to auto")
    check(s.compare_filter == "nearest", "a non-string filter falls back")
    check(s.outer_tile is None, "a non-int outer tile falls back to auto")
    check(s.calibration.rates == {}, "a non-dict calibration becomes empty")
    check(st.Settings.from_dict(None).model_id == catalog.DEFAULT_MODEL_ID,
          "from_dict(None) does not raise")
    check(st.Settings.from_dict([1, 2, 3]).scale > 0, "from_dict(list) does not raise")


def test_clamping():
    print("\nvalues are clamped, not rejected")
    check(st.Settings.from_dict({"engine_tile": 999999}).engine_tile == st.MAX_ENGINE_TILE,
          "an absurd engine tile clamps to the maximum")
    check(st.Settings.from_dict({"engine_tile": 4}).engine_tile == st.MIN_ENGINE_TILE,
          "an engine tile the binary would reject clamps up to 32")
    check(st.Settings.from_dict({"engine_tile": 0}).engine_tile == 0,
          "0 stays 0 — it means 'let the engine choose'")
    check(st.Settings.from_dict({"outer_tile": 0}).outer_tile == 0,
          "outer tile 0 stays 0 — it means single pass")


def test_scale_must_match_model():
    print("\nscale is validated against the chosen model")
    s = st.Settings.from_dict({"model_id": "ultrasharp-4x", "scale": 2})
    check(s.scale == 4, "2x on a 4x-only model is corrected to 4")
    s = st.Settings.from_dict({"model_id": "realesr-animevideov3", "scale": 3})
    check(s.scale == 3, "3x is kept for the model that supports it")
    s = st.Settings.from_dict({"model_id": "realesr-animevideov3", "scale": 7})
    check(s.scale == 4, "an unsupported scale falls back to the model default")


def test_output_scale():
    print("\noutput scale")
    check(st.Settings.from_dict({"scale": 4, "output_scale": 2}).output_scale == 2,
          "asking for 2x out of a 4x run is allowed")
    check(st.Settings.from_dict({"scale": 4, "output_scale": 8}).output_scale is None,
          "an output scale above the engine scale is dropped (we cannot upsample)")
    check(st.Settings.from_dict({"scale": 4, "output_scale": "2"}).output_scale is None,
          "a non-int output scale is dropped")


def test_unknown_keys_survive():
    print("\nforward compatibility")
    d = st.Settings.from_dict({"model_id": "ultrasharp-4x",
                               "future_option": {"a": 1}}).to_dict()
    check(d.get("future_option") == {"a": 1},
          "a key this version does not know is preserved on the round trip")


def test_calibration_blending():
    print("\ncalibration")
    c = st.Calibration()
    check(c.get("m", 4) is None, "an unmeasured model has no rate")
    c.record("m", 4, 100.0)
    check(c.get("m", 4) == 100.0, "the first measurement is taken as-is")
    c.record("m", 4, 50.0)
    blended = c.get("m", 4)
    check(50.0 < blended < 100.0,
          f"a later measurement blends rather than replaces ({blended})")
    c.record("m", 4, 0)
    check(c.get("m", 4) == blended, "a zero measurement is ignored")
    c.record("m", 4, -5)
    check(c.get("m", 4) == blended, "a negative measurement is ignored")
    check(c.get("m", 2) is None, "rates are keyed by scale as well as model")
    check(st.Calibration.from_dict({"m@4": "abc"}).rates == {},
          "an unparseable rate is dropped")


def test_throughput_is_size_independent():
    """The bug this guards: `elapsed / mpx` is not a property of the model.

    Every run pays a fixed startup cost, so dividing total time by megapixels
    gives a figure that changes with image size — and small runs then poison the
    stored calibration for large ones. These are four real timings of
    `upscayl-standard-4x` on one machine; the corrected figure must agree across
    all of them far better than the naive one.
    """
    print("\nthroughput is a model property, not an image-size property")
    model = catalog.get("upscayl-standard-4x")
    runs = [(0.262, 15.2), (1.048, 49.0), (0.130, 9.1), (2.074, 91.9)]

    naive = [el / mp for mp, el in runs]
    fixed = [runner.throughput(model, el, mp) for mp, el in runs]
    spread = lambda v: (max(v) - min(v)) / (sum(v) / len(v))
    check(spread(naive) > 0.4,
          f"the naive elapsed/mpx really does vary with size ({spread(naive):.0%})")
    check(spread(fixed) < 0.15,
          f"startup-corrected throughput is stable across sizes ({spread(fixed):.0%})")

    # And the round trip: the prior should predict the runs it was fitted to.
    for mp, el in runs:
        side = int((mp * 1e6) ** 0.5)
        predicted = runner.estimate_seconds(model, side, side)
        check(abs(predicted - el) < max(3.0, 0.15 * el),
              f"{side}x{side}: predicted {predicted:.0f}s vs measured {el:.0f}s")

    check(runner.throughput(model, 10.0, 0) == 0.0, "a zero-megapixel run reports 0")
    check(runner.throughput(model, 0.1, 1.0) > 0,
          "a run faster than its startup prior still reports a positive rate")


def test_estimate_uses_calibration():
    print("\nthe estimate prefers a measured rate over the prior")
    model = catalog.get("upscayl-standard-4x")
    prior = runner.estimate_seconds(model, 1000, 1000)
    faster = runner.estimate_seconds(model, 1000, 1000, sec_per_mpx=10.0)
    check(faster < prior, "a measured rate of 10 s/MP predicts less than the prior")
    check(abs(faster - (model.startup_s + 10.0)) < 0.01,
          "the estimate is startup + rate * megapixels")
    check(runner.estimate_seconds(model, 1000, 1000, sec_per_mpx=0) == prior,
          "a zero rate falls back to the prior instead of predicting instant")


def test_disk_round_trip():
    print("\npersistence to disk")
    s = st.Settings(model_id="remacri-4x", engine_tile=64)
    s.calibration.record("remacri-4x", 4, 51.2)
    st.save(s)
    check(paths.settings_file().is_file(), f"wrote {paths.settings_file()}")
    back = st.load()
    check(back.model_id == "remacri-4x" and back.engine_tile == 64,
          "settings reload from disk")
    check(back.calibration.get("remacri-4x", 4) is not None, "calibration reloads")

    paths.settings_file().write_text("{ this is not json")
    check(st.load().model_id == catalog.DEFAULT_MODEL_ID,
          "a corrupt settings file loads as defaults instead of crashing")


def main():
    for fn in (test_defaults, test_round_trip, test_garbage_tolerance, test_clamping,
               test_scale_must_match_model, test_output_scale, test_unknown_keys_survive,
               test_calibration_blending, test_throughput_is_size_independent,
               test_estimate_uses_calibration, test_disk_round_trip):
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
