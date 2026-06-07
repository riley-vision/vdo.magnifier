# RapidOCR engine (ONNX Runtime Web · WebGPU)

A self-contained, GPU-accelerated OCR engine for **VDO Magnifier**, loaded lazily only
when the user selects it. The main `vdo-magnifier.html` never grows — all the logic lives
in `rapidocr.js` here, and the runtime + models live in this folder.

## What's already here

```
rapidocr/
├── rapidocr.js                    ← the engine (detect → classify → recognize), provided
├── ort/                           ← onnxruntime-web runtime, provided
│   ├── ort.webgpu.min.js
│   ├── ort-wasm-simd-threaded.jsep.mjs
│   └── ort-wasm-simd-threaded.jsep.wasm   (~22 MB; the WebGPU/WASM kernels)
├── models/                        ← ⚠️ YOU ADD THESE (see below)
└── README.md
```

## What you need to add — `models/`

Drop these **four files** into `rapidocr/models/` with **exactly these names** (rename after
downloading). The app probes for them and only offers RapidOCR once they're present.

| Filename (required) | What it is |
|---|---|
| `det.onnx` | text **detection** model (finds text regions) |
| `rec.onnx` | text **recognition** model (reads each region) |
| `dict.txt` | the recognition **dictionary** — **must match `rec.onnx`** |
| `cls.onnx` | *(optional)* angle classifier; rotates upside-down lines. Omit to skip. |

> ⚠️ **The dict must match the rec model.** The decoder maps each output class to a line in
> `dict.txt` (index 0 is the CTC blank, so line 1 = class 1, …, plus a trailing space). If you
> pair the wrong dict with a rec model, recognized text comes out garbled — `rapidocr.js`
> logs a console warning when the model's class count ≠ dictionary length.

### Recommended source — RapidOCR model zoo on Hugging Face

[`SWHL/RapidOCR`](https://huggingface.co/SWHL/RapidOCR/tree/main) (and PaddleOCR's model list).

**Option A — English + Chinese (recommended default, "just works" for English too):**
| Download | Rename to |
|---|---|
| `ch_PP-OCRv4_det_infer.onnx` (~4.7 MB) | `det.onnx` |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` (~1.4 MB) | `cls.onnx` |
| `ch_PP-OCRv4_rec_infer.onnx` (~10.8 MB) | `rec.onnx` |
| `ppocr_keys_v1.txt` (6623 lines) | `dict.txt` |

**Option B — English only (smaller / faster):**
| Download | Rename to |
|---|---|
| `en_PP-OCRv3_det_infer.onnx` (~2.4 MB) | `det.onnx` |
| `en_PP-OCRv4_rec_infer.onnx` (English) | `rec.onnx` |
| `en_dict.txt` (~96 lines) | `dict.txt` |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` *(optional)* | `cls.onnx` |

The detection model is language-agnostic; the **rec model + dict** are what determine the
language(s). Use the larger `*_server_*` rec model for maximum accuracy at the cost of size.

## How it loads (lazy, like Tesseract)

- Nothing is fetched until the user picks **RapidOCR** as the OCR engine.
- Then the app injects `rapidocr.js`, which loads `ort/ort.webgpu.min.js`, points the wasm
  paths at `ort/`, runs single-threaded (no COOP/COEP needed), and creates the 3 sessions.
- Execution provider: **WebGPU** when `navigator.gpu` exists, automatic **WASM** fallback.

## Licensing

| Component | License | Where |
|---|---|---|
| **onnxruntime-web** (the runtime in `ort/`) | **MIT** © Microsoft Corporation | `ort/LICENSE` + `ort/ThirdPartyNotices.txt` (bundled) |
| **`rapidocr.js`** (this engine) | Part of VDO Magnifier — original code; implements the public PP-OCR pre/post-processing conventions | this folder |
| **The models you add** (`models/*.onnx`, `dict.txt`) | **Apache 2.0** (PaddleOCR / RapidOCR) | keep the upstream `LICENSE`/`NOTICE` with them — see [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / [RapidOCR](https://github.com/RapidAI/RapidOCR) |

The onnxruntime-web MIT license requires its copyright notice to ship with the redistributed
binaries — that's why `ort/LICENSE` and `ort/ThirdPartyNotices.txt` are kept alongside the
runtime. The PP-OCR models are Apache-2.0; if you redistribute them, retain their NOTICE.

## Bounding boxes / coordinates

RapidOCR **does** produce bounding boxes — text detection is the whole first stage — but at
the **line/region level**, not per-word like Tesseract. `recognize()` returns them in
`res.lines`, each `{ text, x0, y0, x1, y1, score }` in **original-image pixel coordinates**:

```js
const res = await RapidOCR.recognize(canvas);
// res.text  → reading-ordered text
// res.lines → [{ text, x0, y0, x1, y1, score }, …]  (line-level detection boxes)
// res.words → []   (PP-OCR's CTC recognizer doesn't emit per-word x-positions)
```

The app uses these for **line-level highlighting**: `runRapidOcr()` explodes each line into its
words (every word carrying that line's box), so the existing highlighter lights up the **current
line** as it reads and jumps line-to-line. It's not word-precise like Tesseract, but RapidOCR
gets a "following" highlight. (Boxes are in the OCR-canvas coordinate space, same as Tesseract,
so the existing overlay mapping applies unchanged.)

## Notes / limitations

- **Line-level highlight, not per-word.** Detection yields line regions, so the highlight
  follows the current *line* as it reads (derived from `res.lines`), rather than the word-precise
  box Tesseract gives. True per-word boxes aren't available from PP-OCR's CTC recognizer.
- Detection post-processing uses **axis-aligned connected-component boxes** tuned for the
  horizontal, high-contrast screen text this tool reads — not full rotated-rect DBNet. Heavily
  rotated/curved text isn't a target here.
- Tunables live in `OPTS` at the top of `rapidocr.js`. Most important: **`detPadHeightRatio`**
  (default `1.0`) — how much each detection box is padded vertically before recognition.
  Detection fires on the text *core*, so without enough padding the crop clips glyphs and the
  recognizer drops spaces / misreads letters; `1.0` (≈ full glyph height + margin) was tuned to
  fix exactly that. Lower it only if dense/tightly-spaced lines start bleeding into each other.
  Others: `detThresh`, `detBoxThresh`, `detPadWidthRatio`, `recImgH`.

## Public API (`window.RapidOCR`)

```js
await RapidOCR.init({ base: '<url>/rapidocr/', onStatus });  // loads ort + models + dict
const { text, lines, backend } = await RapidOCR.recognize(canvas);
RapidOCR.isReady();      // bool
RapidOCR.getBackend();   // 'webgpu (wasm fallback)' | 'wasm'
```
