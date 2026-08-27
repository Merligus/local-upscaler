"""The model catalog: what can be downloaded, and what each model is good at.

Pure data plus a little naming logic. No I/O, no Qt — so `tests/test_catalog.py`
can check the whole table without a network or a display.

Models are **not** vendored into this repository. Each entry names a file pair in
one of Upscayl's two public repositories, and `engine/fetch.py` downloads the
pair on demand into `paths.models_dir()`. Two consequences worth stating:

* Nothing here is redistributed, so the per-model `licence` field is
  informational only. Several of these are non-commercial (`CC-BY-NC-SA-4.0`);
  the authoritative statement for any given model is its openmodeldb.info page,
  which `url` points at. The UI shows the field so a user picking a model for
  commercial work is not surprised.
* `param_bytes`/`bin_bytes` are the exact sizes read from the GitHub API. They
  are what makes the download progress bar determinate, and `fetch.py` treats a
  size mismatch as a failed download — which is the cheap way to catch a
  truncated transfer or an HTML error page saved under a `.bin` name.

**The `realesr-animevideov3` special case.** The ncnn binary builds its parameter
path as `<dir>/<name>.param` for every model *except* that one exact string,
where it instead builds `<dir>/<name>-x<scale>.param`. That is hardcoded in
upstream's `main.cpp`, not configurable, and it is why this entry carries three
scales and a different `files()` result while every other entry carries one. It
is also the only model here that can do 2x and 3x at all.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Where each repository's model directory lives, as a raw-content URL prefix.
_UPSCAYL = "https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/"
_CUSTOM = "https://raw.githubusercontent.com/upscayl/custom-models/main/models/"

#: The one model name the binary rewrites to `<name>-x<scale>`. See the docstring.
SCALED_FILENAME_MODEL = "realesr-animevideov3"


@dataclass(frozen=True)
class Model:
    """One selectable upscaling model."""

    #: Passed to the binary as `-n`, and the basename of the file pair.
    id: str
    #: Shown in the combo box.
    label: str
    #: Scales this model can produce. Length > 1 enables the scale selector.
    scales: tuple[int, ...]
    #: One line of "what is this for", shown under the combo box.
    blurb: str
    author: str
    licence: str
    #: openmodeldb.info page, or the upstream repository.
    url: str
    #: Raw-content prefix the file pair is fetched from.
    base_url: str
    param_bytes: int
    bin_bytes: int
    #: Seconds per input megapixel — a *prior*, replaced by measurement after
    #: the first real run. See `settings.Calibration`.
    sec_per_mpx: float
    #: Fixed cost of a run: Vulkan init, shader compilation and model load. It
    #: is paid once per run regardless of image size, which is precisely why
    #: `runner` sends every tile through a single subprocess invocation.
    startup_s: float

    @property
    def download_bytes(self) -> int:
        """Total bytes for one scale's file pair."""
        return self.param_bytes + self.bin_bytes

    def stem(self, scale: int) -> str:
        """Basename of the file pair for `scale`, without extension."""
        if self.id == SCALED_FILENAME_MODEL:
            return f"{self.id}-x{scale}"
        return self.id

    def filenames(self, scale: int) -> tuple[str, str]:
        """The (`.param`, `.bin`) filenames this model needs at `scale`."""
        stem = self.stem(scale)
        return f"{stem}.param", f"{stem}.bin"

    def default_scale(self) -> int:
        return 4 if 4 in self.scales else self.scales[0]


# Sizes below are exact byte counts from the GitHub contents API, not estimates.
#
# The timing figures are anchored to two measurements taken on the development
# machine (GTX 1050 Ti, 4 GB, Pascal) at `-t 128`, fitting wall time against
# 512x512 and 1024x1024 inputs at 4x:
#
#     ultrasharp-4x     (33 MB RRDBNet-23)   49.4 s/MP,  12.8 s startup
#     upscayl-lite-4x   (2.4 MB SRVGGNet)     8.7 s/MP,   1.7 s startup
#
# Everything else is scaled from those two by architecture and parameter count,
# so treat the other rows as informed estimates rather than measurements. They
# only seed the ETA; `settings.Calibration` overwrites each one with the real
# throughput as soon as the user has actually run that model once.
MODELS: tuple[Model, ...] = (
    Model(
        id="upscayl-standard-4x", label="Upscayl Standard", scales=(4,),
        blurb="General-purpose photos. Real-ESRGAN x4plus — the safe default.",
        author="Xintao Wang et al.", licence="BSD-3-Clause",
        url="https://github.com/xinntao/Real-ESRGAN",
        base_url=_UPSCAYL, param_bytes=116029, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
    Model(
        id="ultrasharp-4x", label="4x UltraSharp", scales=(4,),
        blurb="Crisp detail and texture. Strongest on JPEG-compressed sources.",
        author="Kim2091", licence="CC-BY-NC-SA-4.0",
        url="https://openmodeldb.info/models/4x-UltraSharp",
        base_url=_UPSCAYL, param_bytes=116029, bin_bytes=33424520, sec_per_mpx=49.4, startup_s=12.8,
    ),
    Model(
        id="remacri-4x", label="4x Remacri", scales=(4,),
        blurb="Photographic detail without the plastic over-smoothing.",
        author="FoolhardyVEVO", licence="CC-BY-NC-SA-4.0",
        url="https://openmodeldb.info/models/4x-Remacri",
        base_url=_UPSCAYL, param_bytes=140295, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
    Model(
        id="ultramix-balanced-4x", label="4x UltraMix Balanced", scales=(4,),
        blurb="A gentler blend. Less aggressive than UltraSharp on faces and skin.",
        author="Kim2091", licence="CC-BY-NC-SA-4.0",
        url="https://openmodeldb.info/models/4x-UltraMix-Balanced",
        base_url=_UPSCAYL, param_bytes=140295, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
    Model(
        id="high-fidelity-4x", label="High Fidelity", scales=(4,),
        blurb="Stays closest to the source. Invents the least new detail.",
        author="Upscayl", licence="see OpenModelDB",
        url="https://openmodeldb.info/",
        base_url=_UPSCAYL, param_bytes=108039, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
    Model(
        id="digital-art-4x", label="Digital Art", scales=(4,),
        blurb="Illustration, flat colour and line art. Not for photographs.",
        author="Upscayl", licence="see OpenModelDB",
        url="https://openmodeldb.info/",
        base_url=_UPSCAYL, param_bytes=30290, bin_bytes=8943500, sec_per_mpx=16.0, startup_s=5.0,
    ),
    Model(
        id="upscayl-lite-4x", label="Upscayl Lite", scales=(4,),
        blurb="Roughly ten times faster, slightly softer. Good for a quick look.",
        author="Upscayl", licence="see OpenModelDB",
        url="https://openmodeldb.info/",
        base_url=_UPSCAYL, param_bytes=5019, bin_bytes=2435272, sec_per_mpx=8.7, startup_s=1.7,
    ),
    Model(
        id="realesr-animevideov3", label="Anime Video v3", scales=(2, 3, 4),
        blurb="Anime and cartoons. Very fast, and the only model here that does 2x and 3x.",
        author="Xintao Wang et al.", licence="BSD-3-Clause",
        url="https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md",
        base_url=_CUSTOM, param_bytes=3173, bin_bytes=1247368, sec_per_mpx=5.0, startup_s=1.5,
    ),
    Model(
        id="4xNomos8kSC", label="4x Nomos8k SC", scales=(4,),
        blurb="Photorealistic, trained on a modern high-resolution photo set.",
        author="Phhofm", licence="CC-BY-4.0",
        url="https://openmodeldb.info/models/4x-Nomos8kSC",
        base_url=_CUSTOM, param_bytes=108039, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
    Model(
        id="4x_NMKD-Siax_200k", label="4x NMKD Siax", scales=(4,),
        blurb="Clean or lightly compressed photos. Strong detail, slower.",
        author="NMKD", licence="CC-BY-NC-SA-4.0",
        url="https://openmodeldb.info/models/4x-NMKD-Siax-CX",
        base_url=_CUSTOM, param_bytes=108037, bin_bytes=66793352, sec_per_mpx=95.0, startup_s=20.0,
    ),
    Model(
        id="4x_NMKD-Superscale-SP_178000_G", label="4x NMKD Superscale", scales=(4,),
        blurb="Artifact-free real-world images. Gentle, slower.",
        author="NMKD", licence="CC-BY-NC-SA-4.0",
        url="https://openmodeldb.info/models/4x-NMKD-Superscale",
        base_url=_CUSTOM, param_bytes=108037, bin_bytes=66793352, sec_per_mpx=95.0, startup_s=20.0,
    ),
    Model(
        id="4xLSDIRplusC", label="4x LSDIR plus C", scales=(4,),
        blurb="High quality on compressed sources. Trained on the LSDIR set.",
        author="Phhofm", licence="CC-BY-4.0",
        url="https://openmodeldb.info/models/4x-LSDIRplusC",
        base_url=_CUSTOM, param_bytes=108039, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
    Model(
        id="4xLSDIRCompactC3", label="4x LSDIR Compact C3", scales=(4,),
        blurb="Compact network: fast, handles compression artifacts well.",
        author="Phhofm", licence="CC-BY-4.0",
        url="https://openmodeldb.info/models/4x-LSDIRCompactC3",
        base_url=_CUSTOM, param_bytes=2767, bin_bytes=1247368, sec_per_mpx=5.0, startup_s=1.5,
    ),
    Model(
        id="4xHFA2k", label="4x HFA2k", scales=(4,),
        blurb="Anime stills and artwork, rather than video frames.",
        author="Phhofm", licence="CC-BY-4.0",
        url="https://openmodeldb.info/models/4x-HFA2k",
        base_url=_CUSTOM, param_bytes=108039, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
    Model(
        id="RealESRGAN_General_x4_v3", label="Real-ESRGAN General v3", scales=(4,),
        blurb="Compact general-purpose net. Fast, marginally softer than Standard.",
        author="Xintao Wang et al.", licence="BSD-3-Clause",
        url="https://github.com/xinntao/Real-ESRGAN",
        base_url=_CUSTOM, param_bytes=5019, bin_bytes=2435272, sec_per_mpx=8.7, startup_s=1.7,
    ),
    Model(
        id="uniscale_restore", label="Uniscale Restore", scales=(4,),
        blurb="Restoration of degraded or noisy originals.",
        author="Kim2091", licence="CC-BY-NC-SA-4.0",
        url="https://openmodeldb.info/",
        base_url=_CUSTOM, param_bytes=108039, bin_bytes=33424520, sec_per_mpx=49.0, startup_s=13.0,
    ),
)

#: Selected when there are no settings yet.
DEFAULT_MODEL_ID = "upscayl-standard-4x"

_BY_ID = {m.id: m for m in MODELS}


def by_id(model_id: str) -> Model | None:
    return _BY_ID.get(model_id)


def get(model_id: str) -> Model:
    """`by_id`, falling back to the default rather than raising.

    A settings file naming a model this version dropped should reset to the
    default, not crash the app on startup.
    """
    return _BY_ID.get(model_id) or _BY_ID[DEFAULT_MODEL_ID]
