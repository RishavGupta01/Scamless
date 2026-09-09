# Scamless — On-Device Multilingual Scam Detector

**Date:** 2026-09-09
**Status:** Approved design, pending implementation plan

## 1. Summary

A privacy-first scam and phishing detection product that runs entirely on the user's device. A small multilingual model (plus a fusion of lightweight specialist signals) detects scam messages, phishing pages, and manipulation tactics across 50+ languages, and an awareness layer teaches users to recognize scams themselves. Ships as a static web app (GitHub Pages) and a Chrome extension sharing one detection core. No backend, no telemetry, nothing leaves the device.

## 2. Product Shape

- **Web app** (GitHub Pages, pure client-side): paste text, paste a URL, or upload an .eml file. Returns a verdict with explanations. The static site doubles as the scam encyclopedia (Learn mode).
- **Chrome extension** (Manifest V3, same core): red banner on phishing pages, inline badges in Gmail/Outlook/WhatsApp Web, right-click context-menu check, popup with verdict details. Fully functional offline after first model load. Model auto-updates via Web Store releases.
- **Privacy stance is the brand:** zero backend, zero telemetry by default, no account. The absence of a server is a feature, not a limitation.

## 3. Detection Taxonomy

Multi-label classification (a message can trigger several categories simultaneously), 16 labels total — 15 scam categories plus "Safe" (absence of all others):

| # | Category |
|---|----------|
| 1 | Phishing (credential harvesting) |
| 2 | Investment / crypto scam |
| 3 | Romance scam |
| 4 | Tech-support scam |
| 5 | Lottery / prize scam |
| 6 | Job / task scam |
| 7 | Government / bank impersonation |
| 8 | OTP request |
| 9 | Payment pressure |
| 10 | Advance-fee (419) |
| 11 | Sextortion |
| 12 | Delivery scam |
| 13 | Account-suspension threat |
| 14 | Malicious link present |
| 15 | Generic spam |

Additionally, **tactic spans** are detected at token level and highlighted in the text: urgency, authority, fear, payment pressure.

Explanations are generated **rule-based** from labels and spans. No generative model is used, which eliminates hallucination risk entirely.

## 4. Model Architecture

- **Backbone:** mMiniLM-L6-v2 (118M params, 50+ languages pre-trained). Two fine-tuned heads:
  - Message-level multi-label classifier
  - Token-level span tagger (tactic spans)
- **URL specialist:** separate small char-level classifier trained on PhishTank/OpenPhish (malicious) vs Tranco top-1M (benign). URLs are not prose and get a dedicated model.
- **Runtime:** ONNX int8 quantized, ~45–60MB, loaded once and cached in the browser. Inference via transformers.js with WebGPU, WASM fallback.
- **Training budget:** full fine-tune in 1–3 hours on a consumer GPU or free Colab T4.

## 5. Intelligence Architecture (Signal Fusion)

A single small model is not the intelligence strategy; fusion of specialists is.

**Two-stage pipeline:**
1. **Stage 1 — fast screening:** message classifier, under 20ms. If risk is below threshold, stop here.
2. **Stage 2 — deep pass** (only when Stage 1 risk exceeds threshold): token-level tactic spans, full URL extraction and URL-model scoring on every link in the text, homoglyph/IDN spoof detection (e.g., Cyrillic "а" in "pаypаl.com"), zero-width character and leetspeak normalization, sender/subject pattern analysis.

**Signal fusion:** all signals (text classifier, URL model, link count, TLD reputation list, urgency-span density, metadata patterns) feed a tiny learned combiner (~100KB gradient-boosted model) producing the final calibrated 0–100 score. This is how a 60MB system outperforms a single big model on this task.

**Calibrated confidence:** temperature scaling so a displayed score of 87 means 87. Low confidence produces "Needs review — here's what to check" rather than a fabricated verdict.

**Adversarial robustness:** training-time augmentation with character noise, transliteration, code-switching, and emoji-obfuscation; a standing adversarial eval suite gates every model release.

## 6. Data Pipeline

- **Public corpora:** SMS Spam Collection, Nazario phishing corpus, SpamAssassin, Enron, PhishTank, OpenPhish, Tranco top-1M, Kaggle scam datasets.
- **Synthetic multilingual expansion:** one-time generation per category across target languages using a frontier LLM (offline, part of the training pipeline only) plus back-translation augmentation. This is how multilingual coverage exists where public datasets do not.
- **Language priority for expansion:**
  - Tier 1 (India): Hindi, Hinglish (code-mixed), Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi
  - Tier 2 (international): Spanish, Portuguese (BR), French, Arabic, Indonesian, Russian, Vietnamese, Swahili, Nigerian English
  - Rationale: scam volume concentration plus model backbone coverage; tier order sets synthetic-generation priority.
- **Hard negatives (false-positive control):** real OTPs, legitimate bank alerts, delivery updates, meeting invites. False positives destroy trust, so precision is optimized as hard as recall.

## 7. Quality Targets ("Industry Beating" Defined)

| Metric | Target |
|--------|--------|
| Macro-F1, English phishing/spam | >= 0.95 |
| Macro-F1, low-resource languages | >= 0.85 (Tier 1 + Tier 2 language list, section 6) |
| False-positive rate on benign corpora | < 1% |
| Verdict latency, WebGPU | < 50ms |
| Verdict latency, WASM | < 200ms |
| Install size | < 60MB |

Every model release passes a CI eval harness with per-category regression gates; model quality cannot silently degrade.

## 8. Extension Architecture

- Manifest V3. The model runs in an **offscreen document** (service workers cannot perform heavy inference).
- Content scripts per supported site (Gmail, Outlook, WhatsApp Web) plus a generic fallback.
- Popup contains: verdict card, sensitivity slider, "this was wrong" correction button.
- Page banners are calm and dismissible, never alarmist.
- No analytics by default. Local-only detection history (see Awareness layer).

## 9. Awareness Layer (User Education as a First-Class Feature)

Detection tells users *this one* is a scam; awareness ensures they never fall for the *next* one.

- **Graduated risk meter:** every scan returns a 0–100 risk score with three bands: Safe / Suspicious / Dangerous. Users learn the spectrum, calibrating intuition over time. Risk is rendered with color plus label plus icon, never emoji.
- **"Why" panel on every detection:** each highlighted tactic span carries a plain-language tooltip (e.g., "Creates false urgency so you act before thinking"). A category card explains the scam pattern in two lines.
- **"What to do now" playbook per category:** e.g., OTP scam: "No bank ever asks for OTP. Don't share it, don't click." Phishing link: "Don't click. Type the real address yourself." One-tap copy of a report template with links to FTC / national cybercrime portals / platform reporting.
- **Learn mode:** browsable scam encyclopedia on the same static site — all categories, redacted real examples, interactive "spot the 5 red flags" demos.
- **Micro-lessons:** after each detection, a rotating one-line tip ("Scammers pretend to be authority figures — verify through official channels").
- **Personal detection history:** local-only, exportable; users see their own protection pattern ("3 phishing links blocked this month").

## 10. UI/UX Direction

- **Aesthetic:** dark-mode-first, minimal, typography-led, high contrast, generous whitespace, subtle micro-animations. Premium security product feel, not hobby tool.
- **Three surfaces, one design language:** web app (hero scan bar, animated risk meter, Why panel, playbook), extension popup (compact verdict card, same meter), page banner (calm, dismissible).
- **No-emoji rule (absolute):** zero emojis in UI, copy, code, docs, or commit messages. Icons come from an SVG set (Lucide). Risk levels use color + label + icon.
- **Accessibility:** WCAG AA, keyboard-first, screen-reader labels on all verdicts, i18n-ready strings from day one.
- **Design system:** a design-consultation pass producing DESIGN.md (fonts, palette, components) happens before any UI code is written during implementation.

## 11. Testing & Evaluation

- CI eval harness with per-category regression gates on every model release.
- Span-alignment unit tests (highlight offsets must match source text exactly).
- Playwright E2E for the web app.
- Extension E2E via loaded CRX in Chromium.
- Adversarial suite gates releases (character noise, transliteration, code-switching attacks).

## 12. Phases

1. **Data pipeline + eval harness** — reproducible dataset build, baseline metrics.
2. **Model v1 + web app** — English-first classifier, score fusion, GitHub Pages deployment.
3. **Multilingual expansion + tactic spans** — synthetic data, span tagger, Why panel.
4. **Chrome extension** — offscreen inference, content scripts, popup, banners.
5. **Feedback loop** — local corrections export, adversarial suite hardening, model iteration.

## 13. Explicit Non-Goals (v1)

- No native desktop app (Phase 2+ candidate).
- No mobile apps (Android/iOS on-device SMS filtering is a separate future product).
- No backend, no accounts, no cloud sync.
- No generative model in the detection path.
