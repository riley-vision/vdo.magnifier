/* ============================================================================
 * RapidOCR engine for VDO Magnifier
 * PP-OCR pipeline (text detection → angle classification → recognition) running
 * on onnxruntime-web with the WebGPU execution provider (automatic WASM fallback).
 *
 * Self-contained: the main app only injects THIS file and calls window.RapidOCR.
 * It loads its own onnxruntime-web from ./ort/ and the models you provide in
 * ./models/ (see README.md). Nothing here runs until init() is called, so the
 * ~22 MB runtime + your models are only fetched when the RapidOCR engine is used.
 *
 * Pipeline / preprocessing follows the PaddleOCR (PP-OCR) conventions that the
 * RapidOCR ONNX models are exported with:
 *   - Detection: resize (limit long side, snap to /32), normalise with ImageNet
 *     mean/std, run DBNet, threshold the probability map, take connected
 *     components, score + "unclip"-expand each box.
 *   - Classification (optional): 3×48×192, (x/255-0.5)/0.5, flip 180° if argmax==1.
 *   - Recognition: height 48, (x/255-0.5)/0.5, CTC greedy decode against the dict
 *     (index 0 = CTC blank; remaining indices = dict entries, trailing space).
 *
 * NOTE: detection post-processing here uses axis-aligned connected-component boxes
 * (not rotated min-area rects + pyclipper). That's deliberate — this tool reads
 * horizontal, high-contrast screen/document text, where axis-aligned line boxes are
 * accurate and far simpler/faster than the full DBNet contour machinery.
 * ==========================================================================*/
(function () {
  'use strict';

  const OPTS = {
    base: '',                 // set by init(); folder containing this file (has ort/ and models/)
    ortDir: 'ort/',
    detModel: 'models/det.onnx',
    clsModel: 'models/cls.onnx',   // optional — skipped if absent
    recModel: 'models/rec.onnx',
    dictFile: 'models/dict.txt',
    // detection
    detLimitSideLen: 960,     // longest side is scaled to ≤ this, then snapped to a /32 grid
    detThresh: 0.3,           // probability-map threshold
    detBoxThresh: 0.5,        // min mean-probability for a region to be kept
    // Box padding to recover full glyph height + the margin the recognizer expects. DB fires
    // on the text CORE, so the raw box is thinner than the glyphs and clips ascenders/
    // descenders — and a heavily up-scaled clipped crop makes the recognizer drop spaces and
    // misread letters. Pad proportional to box HEIGHT (≈ font size): generous vertical, modest
    // horizontal. (Padding proportional to box *width* over-expands long lines sideways while
    // barely adding height — backwards for wide single-line text.)
    detPadHeightRatio: 1.0,   // vertical pad each side ≈ 1.0 × box height (tuned: full glyphs + margin)
    detPadWidthRatio: 0.3,    // horizontal pad each side ≈ 0.3 × box height
    detMinSize: 3,            // drop boxes whose short side (map px) is smaller than this
    maxBoxes: 256,            // safety cap on regions per frame
    // recognition
    recImgH: 48,
    recImgMaxW: 1600,         // generous: screen text lines can be wide
    // classification
    clsThresh: 0.9,
  };

  let _ort = null;            // the onnxruntime-web namespace
  let _sessions = { det: null, cls: null, rec: null };
  let _chars = null;          // CTC label list: ['<blank>', ...dict, ' ']
  let _ready = false;
  let _initPromise = null;
  let _backend = 'unknown';
  let _warnedClassCount = false;

  // ---- small DOM/util helpers ------------------------------------------------
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('Failed to load ' + src));
      document.head.appendChild(s);
    });
  }
  function makeCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = Math.max(1, w | 0); c.height = Math.max(1, h | 0);
    return c;
  }
  function ctx2d(c) { return c.getContext('2d', { willReadFrequently: true }); }

  // ---- onnxruntime bootstrap -------------------------------------------------
  async function ensureOrt() {
    if (window.ort) { _ort = window.ort; return; }
    await loadScript(OPTS.base + OPTS.ortDir + 'ort.webgpu.min.js');
    if (!window.ort) throw new Error('onnxruntime-web did not register a global `ort`');
    _ort = window.ort;
    // Point the wasm/jsep artifacts at our local folder; single-threaded avoids the
    // cross-origin-isolation (COOP/COEP) requirement of the threaded build — WebGPU
    // does the heavy lifting anyway.
    _ort.env.wasm.wasmPaths = OPTS.base + OPTS.ortDir;
    _ort.env.wasm.numThreads = 1;
  }

  async function createSession(url, optional) {
    const tryEps = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    try {
      const s = await _ort.InferenceSession.create(url, {
        executionProviders: tryEps,
        graphOptimizationLevel: 'all',
      });
      _backend = navigator.gpu ? 'webgpu (wasm fallback)' : 'wasm';
      return s;
    } catch (e) {
      if (navigator.gpu) {
        // WebGPU EP failed for this model — retry on wasm before giving up.
        try {
          const s = await _ort.InferenceSession.create(url, {
            executionProviders: ['wasm'], graphOptimizationLevel: 'all',
          });
          _backend = 'wasm';
          return s;
        } catch (e2) { if (optional) return null; throw e2; }
      }
      if (optional) return null;
      throw e;
    }
  }

  async function fetchDict(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('dict fetch ' + res.status + ' (' + url + ')');
    const lines = (await res.text()).replace(/\r/g, '').split('\n');
    // PP-OCR keeps trailing newline; drop only a final empty entry, keep interior ones.
    if (lines.length && lines[lines.length - 1] === '') lines.pop();
    // CTC label list: index 0 is blank; dict entries follow; PP-OCR appends a space.
    return ['<blank>'].concat(lines).concat([' ']);
  }

  // ---- detection -------------------------------------------------------------
  function preprocessDet(srcCanvas) {
    const w0 = srcCanvas.width, h0 = srcCanvas.height;
    let ratio = 1;
    const maxSide = Math.max(w0, h0);
    if (maxSide > OPTS.detLimitSideLen) ratio = OPTS.detLimitSideLen / maxSide;
    let rw = Math.max(32, Math.round((w0 * ratio) / 32) * 32);
    let rh = Math.max(32, Math.round((h0 * ratio) / 32) * 32);

    const c = makeCanvas(rw, rh);
    ctx2d(c).drawImage(srcCanvas, 0, 0, rw, rh);
    const data = ctx2d(c).getImageData(0, 0, rw, rh).data;

    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    const plane = rw * rh;
    const out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      out[p]             = (data[i]     / 255 - mean[0]) / std[0];
      out[plane + p]     = (data[i + 1] / 255 - mean[1]) / std[1];
      out[2 * plane + p] = (data[i + 2] / 255 - mean[2]) / std[2];
    }
    return {
      tensor: new _ort.Tensor('float32', out, [1, 3, rh, rw]),
      scaleW: w0 / rw, scaleH: h0 / rh, rw, rh,
    };
  }

  // 4-connected component bounding boxes over a binary map.
  function connectedComponentBoxes(bin, w, h, minPixels) {
    const visited = new Uint8Array(w * h);
    const boxes = [];
    const stack = [];
    for (let s = 0; s < bin.length; s++) {
      if (!bin[s] || visited[s]) continue;
      let minx = s % w, maxx = minx, miny = (s / w) | 0, maxy = miny, count = 0;
      stack.length = 0; stack.push(s); visited[s] = 1;
      while (stack.length) {
        const idx = stack.pop();
        const x = idx % w, y = (idx / w) | 0;
        count++;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        if (x > 0)     { const n = idx - 1; if (bin[n] && !visited[n]) { visited[n] = 1; stack.push(n); } }
        if (x < w - 1) { const n = idx + 1; if (bin[n] && !visited[n]) { visited[n] = 1; stack.push(n); } }
        if (y > 0)     { const n = idx - w; if (bin[n] && !visited[n]) { visited[n] = 1; stack.push(n); } }
        if (y < h - 1) { const n = idx + w; if (bin[n] && !visited[n]) { visited[n] = 1; stack.push(n); } }
      }
      if (count >= (minPixels || 1)) boxes.push({ x0: minx, y0: miny, x1: maxx, y1: maxy });
    }
    return boxes;
  }

  function detPostprocess(prob, mw, mh, scaleW, scaleH) {
    const bin = new Uint8Array(mw * mh);
    for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > OPTS.detThresh ? 1 : 0;
    const ccBoxes = connectedComponentBoxes(bin, mw, mh, 4);
    const out = [];
    for (const b of ccBoxes) {
      const bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1;
      if (Math.min(bw, bh) < OPTS.detMinSize) continue;
      // mean probability inside the box → region score
      let sum = 0, cnt = 0;
      for (let y = b.y0; y <= b.y1; y++) {
        const row = y * mw;
        for (let x = b.x0; x <= b.x1; x++) { sum += prob[row + x]; cnt++; }
      }
      if (!cnt || sum / cnt < OPTS.detBoxThresh) continue;
      // Map the core box to original-image pixels, then pad proportional to its HEIGHT.
      const ix0 = Math.round(b.x0 * scaleW), iy0 = Math.round(b.y0 * scaleH);
      const ix1 = Math.round(b.x1 * scaleW), iy1 = Math.round(b.y1 * scaleH);
      const ih = Math.max(1, iy1 - iy0);
      const padY = Math.round(ih * OPTS.detPadHeightRatio);
      const padX = Math.round(ih * OPTS.detPadWidthRatio);
      out.push({
        x0: Math.max(0, ix0 - padX),
        y0: Math.max(0, iy0 - padY),
        x1: ix1 + padX,
        y1: iy1 + padY,
        score: sum / cnt,
      });
    }
    return out;
  }

  // ---- classification (optional) --------------------------------------------
  async function maybeRotate180(crop) {
    if (!_sessions.cls) return crop;
    const H = 48, W = 192;
    const c = makeCanvas(W, H);
    ctx2d(c).drawImage(crop, 0, 0, W, H);
    const data = ctx2d(c).getImageData(0, 0, W, H).data;
    const plane = W * H, out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      out[p]             = (data[i]     / 255 - 0.5) / 0.5;
      out[plane + p]     = (data[i + 1] / 255 - 0.5) / 0.5;
      out[2 * plane + p] = (data[i + 2] / 255 - 0.5) / 0.5;
    }
    const t = new _ort.Tensor('float32', out, [1, 3, H, W]);
    const r = await _sessions.cls.run({ [_sessions.cls.inputNames[0]]: t });
    const o = r[_sessions.cls.outputNames[0]].data; // [p0, p180]
    if (o.length >= 2 && o[1] > o[0] && o[1] > OPTS.clsThresh) {
      const rc = makeCanvas(crop.width, crop.height);
      const rx = ctx2d(rc);
      rx.translate(crop.width, crop.height); rx.rotate(Math.PI);
      rx.drawImage(crop, 0, 0);
      return rc;
    }
    return crop;
  }

  // ---- recognition -----------------------------------------------------------
  function ctcDecode(tensor) {
    const dims = tensor.dims;            // [1, T, C]
    const T = dims[1], C = dims[2];
    const d = tensor.data;
    if (!_warnedClassCount && _chars && C !== _chars.length) {
      _warnedClassCount = true;
      console.warn('[RapidOCR] rec model outputs ' + C + ' classes but the dictionary maps ' +
        _chars.length + '. Text will be garbled — the dict.txt must match the rec model.');
    }
    let text = '', prev = -1, sSum = 0, sCnt = 0;
    for (let t = 0; t < T; t++) {
      const off = t * C;
      let best = 0, bestVal = -Infinity;
      for (let c = 0; c < C; c++) { const v = d[off + c]; if (v > bestVal) { bestVal = v; best = c; } }
      if (best !== 0 && best !== prev) {
        const ch = _chars ? _chars[best] : undefined;
        if (ch !== undefined && ch !== '<blank>') { text += ch; sSum += bestVal; sCnt++; }
      }
      prev = best;
    }
    return { text, score: sCnt ? sSum / sCnt : 0 };
  }

  async function recognizeCrop(crop) {
    const H = OPTS.recImgH;
    let W = Math.round(H * crop.width / crop.height);
    W = Math.min(OPTS.recImgMaxW, Math.max(4, W));
    const rc = makeCanvas(W, H);
    ctx2d(rc).drawImage(crop, 0, 0, W, H);
    const data = ctx2d(rc).getImageData(0, 0, W, H).data;
    const plane = W * H, out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      out[p]             = (data[i]     / 255 - 0.5) / 0.5;
      out[plane + p]     = (data[i + 1] / 255 - 0.5) / 0.5;
      out[2 * plane + p] = (data[i + 2] / 255 - 0.5) / 0.5;
    }
    const t = new _ort.Tensor('float32', out, [1, 3, H, W]);
    const r = await _sessions.rec.run({ [_sessions.rec.inputNames[0]]: t });
    return ctcDecode(r[_sessions.rec.outputNames[0]]);
  }

  // ---- reading-order + line assembly ----------------------------------------
  function toReadingText(items) {
    // items: [{text, x0,y0,x1,y1, score}] — group into lines by vertical overlap,
    // order lines top→bottom and words left→right within a line.
    const sorted = items.slice().sort((a, b) => (a.y0 - b.y0) || (a.x0 - b.x0));
    const lines = [];
    for (const it of sorted) {
      const cy = (it.y0 + it.y1) / 2;
      let line = null;
      for (let i = lines.length - 1; i >= 0; i--) {
        const L = lines[i];
        const h = L.y1 - L.y0;
        if (cy >= L.y0 - h * 0.4 && cy <= L.y1 + h * 0.4) { line = L; break; }
      }
      if (!line) { line = { items: [], y0: it.y0, y1: it.y1 }; lines.push(line); }
      line.items.push(it);
      line.y0 = Math.min(line.y0, it.y0); line.y1 = Math.max(line.y1, it.y1);
    }
    lines.sort((a, b) => a.y0 - b.y0);
    return lines.map(L =>
      L.items.sort((a, b) => a.x0 - b.x0).map(i => i.text).join(' ').trim()
    ).filter(Boolean).join('\n');
  }

  // ---- public API ------------------------------------------------------------
  async function init(opts) {
    if (_ready) return;
    if (_initPromise) return _initPromise;
    _initPromise = (async () => {
      Object.assign(OPTS, opts || {});
      if (!OPTS.base) throw new Error('RapidOCR.init needs { base } — the /rapidocr/ folder URL');
      if (typeof OPTS.onStatus === 'function') OPTS.onStatus('Loading ONNX runtime…');
      await ensureOrt();

      if (typeof OPTS.onStatus === 'function') OPTS.onStatus('Loading models…');
      const detUrl = OPTS.base + OPTS.detModel;
      const recUrl = OPTS.base + OPTS.recModel;
      const clsUrl = OPTS.base + OPTS.clsModel;
      // det + rec are required; cls is optional.
      _sessions.det = await createSession(detUrl, false);
      _sessions.rec = await createSession(recUrl, false);
      _sessions.cls = await createSession(clsUrl, true);  // null if not present
      _chars = await fetchDict(OPTS.base + OPTS.dictFile);

      _ready = true;
      if (typeof OPTS.onStatus === 'function') OPTS.onStatus('ready');
    })();
    try { await _initPromise; } catch (e) { _initPromise = null; throw e; }
  }

  async function recognize(source) {
    if (!_ready) throw new Error('RapidOCR.recognize called before init()');
    // Normalise input to a canvas.
    let canvas;
    if (source instanceof HTMLCanvasElement) {
      canvas = source;
    } else {
      canvas = makeCanvas(source.naturalWidth || source.width, source.naturalHeight || source.height);
      ctx2d(canvas).drawImage(source, 0, 0);
    }

    const pre = preprocessDet(canvas);
    const detOut = await _sessions.det.run({ [_sessions.det.inputNames[0]]: pre.tensor });
    const probT = detOut[_sessions.det.outputNames[0]];
    const dd = probT.dims;                       // [1,1,H,W]
    const mh = dd[dd.length - 2], mw = dd[dd.length - 1];
    // det output map matches the resized input, so map→original scale = orig/resized.
    let boxes = detPostprocess(probT.data, mw, mh, pre.scaleW, pre.scaleH);
    boxes.sort((a, b) => (a.y0 - b.y0) || (a.x0 - b.x0));
    if (boxes.length > OPTS.maxBoxes) boxes = boxes.slice(0, OPTS.maxBoxes);

    const items = [];
    for (const b of boxes) {
      // Clamp the padded box to the image before cropping (padding can push it out of bounds).
      const bx0 = Math.max(0, Math.min(canvas.width,  b.x0));
      const by0 = Math.max(0, Math.min(canvas.height, b.y0));
      const bx1 = Math.max(0, Math.min(canvas.width,  b.x1));
      const by1 = Math.max(0, Math.min(canvas.height, b.y1));
      const bw = bx1 - bx0, bh = by1 - by0;
      if (bw < 2 || bh < 2) continue;
      let crop = makeCanvas(bw, bh);
      ctx2d(crop).drawImage(canvas, bx0, by0, bw, bh, 0, 0, bw, bh);
      crop = await maybeRotate180(crop);
      const { text, score } = await recognizeCrop(crop);
      const clean = (text || '').trim();
      if (clean) items.push({ text: clean, x0: bx0, y0: by0, x1: bx1, y1: by1, score });
    }

    return {
      engine: 'rapidocr',
      backend: _backend,
      text: toReadingText(items),
      lines: items,
      words: [],   // line-level boxes only; per-word boxes aren't produced (see README)
    };
  }

  window.RapidOCR = {
    init,
    recognize,
    isReady: () => _ready,
    getBackend: () => _backend,
    configure: (o) => Object.assign(OPTS, o || {}),
    // exposed for unit-testing the pure-JS helpers without models present
    _test: { ctcDecode, connectedComponentBoxes, detPostprocess, toReadingText,
             setChars: (c) => { _chars = c; }, OPTS },
  };
})();
