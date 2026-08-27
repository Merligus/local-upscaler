# TODO

Ordered roughly by how much they would be missed.

## 1. An online-API version — deferred by design

Explicitly out of scope for the first version: the whole point was for it to run
locally. The shape it would take:

* A second backend behind the same `engine.runner` interface, so the UI does not
  learn about it. `Runner` is already Qt-free and callback-driven, which is the
  seam to use.
* Candidates: Replicate, fal.ai, or Topaz's API. All are per-image paid.
* Needs: an API-key store (**not** in `settings.json` — that file is
  world-readable and gets committed by accident), an upload progress stage
  alongside the existing download one, and a clear indication in the UI that the
  image is leaving the machine.
* The privacy trade is the real design question, not the code.

## 2. Importing arbitrary `.pth` models

civitai and OpenModelDB mostly publish PyTorch `.pth` files, which this engine
cannot read. Upscayl documents the conversion
(<https://github.com/upscayl/upscayl/wiki/Model-Conversion-Guide>): `.pth` →
ONNX → ncnn `.param`/`.bin`.

Doing it inside the app would mean pulling in torch and onnx purely as a
conversion tool — a 1.3 GB dependency for something used once per model. More
likely: a documented external recipe, plus a "custom models folder" setting so
converted models appear in the picker alongside the catalog.

## 3. Batch / folder mode

The engine already works on directories, and `engine/tiling.py` already stages
many files through one invocation, so the machinery is there. The work is in the
UI: a queue, per-file progress against overall progress, and deciding what to do
when one image in fifty fails.

## 4. Drag and drop

Dropping an image onto the setup page should load it. `SetupPage.load_image` is
already the single entry point, so this is a `dragEnterEvent`/`dropEvent` pair.

## 5. A difference view

A fourth mode next to Original / Upscaled / Compare, showing a heat map of where
the model changed the most. `ImageView` already draws from a source rectangle per
paint, but a difference has to be *computed*, and computing it for a 192 MP
canvas per frame is not viable — it would need a downsampled difference image
built once when the result arrives.

## 6. Face restoration

GFPGAN and CodeFormer have ncnn ports. They are a separate pass over detected
faces rather than another upscaling model, so they do not fit the current
one-model-one-run structure.

## 7. Video

Out of scope, and probably a different app. The engine supports it, but frame
extraction, re-encoding and audio passthrough are most of the work.

---

## Smaller things

* **No automatic CPU fallback.** If the GPU run fails the error names the fix
  (lower the tile, or switch to CPU) but does not retry by itself. Deliberate: a
  silent fallback to CPU could turn a 90-second run into an hour without asking.
* **Multi-GPU** is not exposed. The engine takes `-g 0,1,2`; the settings model
  assumes one device.
* **The progress bar is coarse for small images.** Tile size is chosen to target
  ~16 tiles but clamped to a 384 px minimum, so a 640x360 source gets 2 tiles and
  the bar moves twice. Acceptable, since such a run takes seconds.
* **TTA is offered but untested at length.** It is upstream's `-x` flag; the 8x
  cost estimate is upstream's claim, not measured here.
* **No way to cancel a download** from the CLI, only from the GUI.
* **Catalog sizes are hand-maintained.** `tests/test_catalog_remote.py` checks
  them against the servers, but nothing runs it automatically, so an upstream
  re-export would not be noticed until a download failed.
