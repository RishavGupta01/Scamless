# Chrome Web Store listing

## Checklist before submitting

1. `python training/scripts/generate_icons.py` - creates store icons
2. `python apps/extension/fetch-deps.py` - vendors the inference bundle
3. `python training/scripts/make_store_zip.py` - builds the upload zip
4. Publish the web app first (GitHub Pages) - the privacy policy URL must be
   live: https://rishavgupta01.github.io/Scamless/privacy.html
5. Register a Chrome Web Store developer account (one-time USD 5 fee) at
   https://chrome.google.com/webstore/devconsole
6. Upload `dist/scamless-extension-v<version>.zip`, fill the listing from
   this document, submit for review (typical review: a few days)

## Listing copy

Name: Scamless - On-Device Scam Detector

Short description (132 chars max):
  Check any message for scams instantly - fully on-device. Private,
  multilingual, explainable verdicts. Nothing leaves your browser.

Category: Security
Language: English (detector covers 20+ languages)

Detailed description:
  Scamless checks suspicious messages - SMS forwards, phishing emails,
  fake bank alerts, lottery scams, OTP theft attempts, task-scam offers -
  entirely on your own device.

  WHY IT IS DIFFERENT
  - No account, no uploads, no server. Detection runs in your browser.
  - A trained multilingual model (20+ languages) fused with link analysis,
    homoglyph detection and pressure-language signals.
  - Every verdict is explainable: see the exact scam category, the signals
    that fired, and what to do next.

  FEATURES
  - Right-click any selected text to check it
  - Per-message Check buttons on Gmail, Outlook and WhatsApp Web
  - Popup scanner with sensitivity control
  - Risk badges on the toolbar, color-coded per tab
  - Works offline after one-time model download

  YOUR PRIVACY
  Messages you scan are processed inside your browser and are never sent
  anywhere. Corrections you mark are stored locally and only leave your
  device if you personally export them.

## Permission justifications (required by review)

- contextMenus: adds the "Check selected text for scams" menu entry.
- scripting + activeTab: runs the check on the text the user explicitly
  selected via the keyboard command on the current page.
- offscreen: hosts the detection model (MV3 requires offscreen documents
  for heavy inference; service workers cannot run it).
- storage / unlimitedStorage: caches the 118 MB model and the user's local
  correction history. Nothing is synced or uploaded.
- host_permissions (huggingface.co, cdn.jsdelivr.net): one-time download of
  the model weights and the inference runtime. No user data touches these
  domains.

## Single purpose statement

This extension has a single purpose: detecting scam content in text the user
chooses to check.
