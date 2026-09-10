# Hosting the detection model for the web app

The web app runs the model fully in-browser via transformers.js. It needs the
exported artifact hosted somewhere reachable with CORS enabled - the Hugging
Face Hub is free and works out of the box.

## 1. Prepare the publish directory

From the repo root, with the training environment active and artifacts copied
locally (the Colab run saves them to `MyDrive/scamless/artifacts/model_v1`):

    python training/scripts/prepare_web_model.py --model-dir artifacts/model_v1

This creates `publish/web-model/` containing exactly what the app expects:

    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
    onnx/model_int8.onnx        (118 MB int8 artifact)
    thresholds.json             (calibrated per-label thresholds)

## 2. Upload to the Hugging Face Hub

Create a free account, then a repo named `scamless-model-v1` (or edit
`apps/web/js/config.js` to match whatever name you choose), and run:

    set HF_TOKEN=your_write_token
    python training/scripts/prepare_web_model.py --model-dir artifacts/model_v1 --upload Rishavgupta/scamless-model-v1

The app reads the repo id from `apps/web/js/config.js` - keep them in sync.

## 3. Enable GitHub Pages

The included workflow (`.github/workflows/deploy-pages.yml`) deploys
`apps/web` to GitHub Pages automatically on every push to `main`. Enable
Pages once in the repository settings: Source = GitHub Actions.

## Notes

- The model downloads once per browser (~118 MB) and is then cached; the app
  shows real MB progress during this per the spec's model-loading UX.
- If the model cannot load, the app falls back to clearly-labelled keyword
  demo mode - the site still works offline.
- The parity between the hosted ONNX file and training weights is verified
  during every training run by `onnx_parity_check` in the eval pipeline.
