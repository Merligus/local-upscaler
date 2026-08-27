"""User preferences, and the measured throughput that makes the ETA honest.

Follows soundboard's pattern: a dataclass, a loader that tolerates anything on
disk, and an atomic save. Values are clamped rather than rejected — a
hand-edited engine tile of 4 should become the minimum the engine accepts, not
crash the app on the next run — and unknown keys survive a round trip so a
newer version's settings are not destroyed by an older one.

`Calibration` is the part worth explaining. The catalog ships a `sec_per_mpx`
prior for every model, but a prior is only ever right for the machine it was
measured on. After each successful run the real figure is recorded here, keyed
by model and scale, and every later estimate for that combination uses the
measurement instead. So the first run of a given model shows an approximate ETA
and every run after it shows a good one, with no benchmark step the user has to
know about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import paths
from .engine import catalog

#: `-t`. 128 is comfortable inside 4 GB of VRAM; 512 was measured to exhaust it
#: on the RRDBNet models. 0 lets the engine guess from reported VRAM, which it
#: does optimistically, so it is offered but not the default.
ENGINE_TILE_CHOICES = (0, 64, 128, 192, 256)
MIN_ENGINE_TILE = 32
MAX_ENGINE_TILE = 1024
DEFAULT_ENGINE_TILE = 128

#: Outer tile. None means "size it from the image"; 0 means one single pass with
#: no progress granularity.
OUTER_TILE_CHOICES = (None, 0, 256, 384, 512, 768, 1024)

#: How the original is drawn in the comparison view. Nearest is the default
#: because it shows the source's actual pixels rather than a smoothed guess at
#: them, which is the honest thing to compare against.
FILTERS = ("nearest", "smooth")

#: `-g`. None lets the engine pick a device; -1 forces CPU.
DEVICES = ("auto", "cpu")


@dataclass
class Calibration:
    """Measured seconds per input megapixel, keyed by `<model id>@<scale>`."""

    rates: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def key(model_id: str, scale: int) -> str:
        return f"{model_id}@{scale}"

    def get(self, model_id: str, scale: int) -> float | None:
        v = self.rates.get(self.key(model_id, scale))
        return v if isinstance(v, (int, float)) and v > 0 else None

    def record(self, model_id: str, scale: int, sec_per_mpx: float) -> None:
        """Blend a new measurement into the stored one.

        An exponential average rather than a replacement: one run that happened
        to share the GPU with a game should not throw the estimate off for
        every run after it, and one that got a warm shader cache should not
        make the estimate permanently optimistic.
        """
        if not (sec_per_mpx and sec_per_mpx > 0):
            return
        k = self.key(model_id, scale)
        prev = self.rates.get(k)
        self.rates[k] = (0.6 * prev + 0.4 * sec_per_mpx
                         if isinstance(prev, (int, float)) and prev > 0
                         else float(sec_per_mpx))

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in self.rates.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        rates = {}
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f > 0:
                    rates[str(k)] = f
        return cls(rates=rates)


def _clamp_int(value, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    model_id: str = catalog.DEFAULT_MODEL_ID
    scale: int = 4
    #: None means "whatever the model produces natively".
    output_scale: int | None = None
    engine_tile: int = DEFAULT_ENGINE_TILE
    #: None means auto-size from the image; 0 means a single pass.
    outer_tile: int | None = None
    tta: bool = False
    device: str = "auto"
    #: Overrides engine discovery when set. Empty means "search normally".
    binary_path: str = ""
    compare_filter: str = "nearest"
    last_open_dir: str = ""
    last_save_dir: str = ""
    calibration: Calibration = field(default_factory=Calibration)
    #: Anything this version does not understand, kept for the round trip.
    _extra: dict = field(default_factory=dict, repr=False)

    #: Keys this class owns; everything else on disk is preserved verbatim.
    _KNOWN = ("model_id", "scale", "output_scale", "engine_tile", "outer_tile",
              "tta", "device", "binary_path", "compare_filter",
              "last_open_dir", "last_save_dir", "calibration")

    def model(self) -> catalog.Model:
        return catalog.get(self.model_id)

    def gpu_arg(self) -> int | None:
        """The `-g` value for the chosen device."""
        return -1 if self.device == "cpu" else None

    def to_dict(self) -> dict:
        d = dict(self._extra)
        d.update({
            "model_id": self.model_id,
            "scale": self.scale,
            "output_scale": self.output_scale,
            "engine_tile": self.engine_tile,
            "outer_tile": self.outer_tile,
            "tta": self.tta,
            "device": self.device,
            "binary_path": self.binary_path,
            "compare_filter": self.compare_filter,
            "last_open_dir": self.last_open_dir,
            "last_save_dir": self.last_save_dir,
            "calibration": self.calibration.to_dict(),
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        if not isinstance(d, dict):
            d = {}
        s = cls(_extra={k: v for k, v in d.items() if k not in cls._KNOWN})

        model = catalog.get(str(d.get("model_id", catalog.DEFAULT_MODEL_ID)))
        s.model_id = model.id
        # The scale must be one this model can actually produce; a settings file
        # naming 2x for a 4x-only model would otherwise reach the engine and
        # produce tiles of an unexpected size.
        scale = _clamp_int(d.get("scale", model.default_scale()), 1, 8,
                           model.default_scale())
        s.scale = scale if scale in model.scales else model.default_scale()

        out = d.get("output_scale")
        s.output_scale = (out if isinstance(out, int) and 1 <= out <= s.scale else None)

        s.engine_tile = _clamp_int(d.get("engine_tile", DEFAULT_ENGINE_TILE),
                                   0, MAX_ENGINE_TILE, DEFAULT_ENGINE_TILE)
        if 0 < s.engine_tile < MIN_ENGINE_TILE:
            s.engine_tile = MIN_ENGINE_TILE      # the engine rejects 1..31

        outer = d.get("outer_tile")
        s.outer_tile = (_clamp_int(outer, 0, 4096, 0)
                        if isinstance(outer, int) else None)

        s.tta = bool(d.get("tta", False))
        s.device = d.get("device") if d.get("device") in DEVICES else "auto"
        s.binary_path = str(d.get("binary_path") or "")
        s.compare_filter = (d.get("compare_filter")
                            if d.get("compare_filter") in FILTERS else "nearest")
        s.last_open_dir = str(d.get("last_open_dir") or "")
        s.last_save_dir = str(d.get("last_save_dir") or "")
        s.calibration = Calibration.from_dict(d.get("calibration") or {})
        return s


def load() -> Settings:
    return Settings.from_dict(paths.read_json(paths.settings_file(), {}))


def save(s: Settings) -> None:
    paths.write_json(paths.settings_file(), s.to_dict())
