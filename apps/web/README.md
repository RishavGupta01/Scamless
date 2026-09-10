# Scamless web app

Static, backend-free web app: on-device scam detection via transformers.js
(ONNX Runtime Web, WebGPU/WASM). Deployed to GitHub Pages by
`.github/workflows/deploy-pages.yml` on every push to `main`.

## Surfaces

- `index.html` - scan flow: message input, risk meter (0-100), verdict band,
  rule-based "Why" panel, per-category playbook. Model loads once with real
  MB progress and is cached by the browser afterwards.
- `learn.html` - scam encyclopedia: all 15 categories, red flags, and the
  action to take. Shares `js/categories.js` with the scan flow.
- `js/config.js` - points at the Hugging Face repo hosting the model artifact.

## Model hosting

The app loads the model from the HF repo in `js/config.js`. To publish the
trained artifact there, see `docs/model-hosting.md` (one script does the
staging and upload).

## Demo mode

If the model cannot load (first-visit offline, HF outage), the app falls back
to a keyword heuristic port (`js/heuristic.js`) and labels it clearly in the
engine tag - users are never shown a trained-model verdict that is not one.

## Privacy

No backend, no telemetry, no account. Inference runs in the browser tab;
scan history stays in localStorage and can be cleared with the site data.
