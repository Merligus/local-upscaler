"""Checks every declared file size against the servers we download from.

Run:  python3 tests/test_catalog_remote.py     (needs network; no GPU, no display)

Kept apart from `test_catalog.py`, which is offline and can only prove the
catalog is *self-consistent*. Self-consistency is not the property that matters
here: `fetch` rejects any download whose length does not match the catalog
exactly, so a number that is merely wrong makes a model permanently
un-downloadable, and no amount of local checking can tell.

That is not hypothetical — it is why this file exists. `realesr-animevideov3`
shipped with one size pair covering all three of its scales, but its x4
parameter file is 3077 bytes where x2 and x3 are 3173. Every x4 download failed
with "expected 3173 bytes, got 3077", and x4 is that model's default scale. A
`HEAD` against the real URL would have caught it before anyone selected the
model.

Worth running after editing the catalog, and occasionally regardless: these are
other people's repositories, and a model can be re-exported upstream at any time.
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_upscaler.engine import binary, catalog  # noqa: E402

FAILS = []
TIMEOUT = 30


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def head(url):
    """(status, content-length) for `url`, or (None, None) if unreachable."""
    request = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "local-upscaler-test"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, int(response.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, ValueError, OSError) as e:
        print(f"        (unreachable: {e})")
        return None, None


def test_model_files():
    print("\nevery model file exists and is the size the catalog claims")
    for model in catalog.MODELS:
        for scale in model.scales:
            expected = dict(zip(model.filenames(scale), model.file_sizes(scale)))
            for name, want in expected.items():
                status, got = head(model.base_url + name)
                if status is None:
                    check(False, f"{name}: could not reach the server")
                elif status != 200:
                    check(False, f"{name}: HTTP {status} — the URL has moved")
                else:
                    check(got == want,
                          f"{name}: {got} bytes"
                          + ("" if got == want else f", but the catalog says {want}"))


def test_engine_archive():
    print("\nthe engine archive is still where and what we think")
    status, got = head(binary.ENGINE_URL)
    if status is None:
        check(False, "engine archive: could not reach the server")
        return
    check(status == 200, f"engine archive: HTTP {status}")
    check(got == binary.ENGINE_ZIP_BYTES,
          f"engine archive: {got} bytes"
          + ("" if got == binary.ENGINE_ZIP_BYTES
             else f", but the catalog says {binary.ENGINE_ZIP_BYTES}"))


def main():
    print("Checking the catalog against live servers. This needs network.")
    for fn in (test_model_files, test_engine_archive):
        fn()
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all passed'}")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
