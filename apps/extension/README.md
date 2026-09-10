# Scamless Chrome extension (Manifest V3)

On-device scam detection inside the browser. Same detection core as the web
app; runs in an offscreen document (MV3 service workers cannot host heavy
inference), with weights cached by transformers.js after one-time download.

## Features (v1)

- Right-click any selected text: "Check selected text for scams"
- Popup paste-scan with risk band and per-label verdicts
- Per-message "Check" buttons on Gmail, Outlook and WhatsApp Web
- Sensitivity slider (stricter thresholds, applied on top of calibrated ones)
- "This was wrong" local correction loop - stored on-device, exportable JSON,
  never uploaded. These corrections feed future training runs.

## Setup (developer)

1. Vendor the inference bundle (MV3 CSP forbids remote scripts):

       python apps/extension/fetch-deps.py

2. Host the model artifact on the Hugging Face Hub (see
   `docs/model-hosting.md`) - the offscreen document loads
   `RishavGupta01/scamless-model-v1` on first use and caches it.

3. Load unpacked: chrome://extensions -> Developer mode -> Load unpacked ->
   select `apps/extension/`.

## Architecture

- `background.js` - context menu, offscreen lifecycle, message routing.
- `offscreen/` - the model lives here; answers `{target: "offscreen"}` scans.
- `content/content.js` - verdict toasts (shadow DOM) + per-message buttons.
- `popup/` - paste scan, sensitivity slider, correction loop.

No telemetry. No backend. Scanning works offline after first model load.
