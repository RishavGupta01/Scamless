# Scamless - go-live guide

Everything needed to take the product from repository to live users, in
order. Steps 1-2 take five minutes and make the site live in demo mode.
Steps 3-4 connect the trained model. Steps 5-6 ship the extension.

---

## Step 1: Enable GitHub Pages (2 minutes)

The deploy workflow (`.github/workflows/deploy-pages.yml`) already exists on
main and triggers automatically once Pages is enabled.

1. Open `github.com/RishavGupta01/Scamless` -> **Settings** -> **Pages**
2. Under **Build and deployment**, set **Source** to **GitHub Actions**
3. Go to the **Actions** tab - the "Deploy web app to GitHub Pages" workflow
   runs on every push to main. Wait for the green check.

Result: the site is live at `https://rishavgupta01.github.io/Scamless/`.
It runs in **demo mode** (keyword rules) until the model is published in
step 3 - the engine tag on every scan says which mode is active.

## Step 2: Hugging Face account and write token (3 minutes)

The model artifact (~235-470 MB) is hosted on the Hugging Face Hub: free,
CORS-enabled, no server to run.

1. Sign up at `https://huggingface.co` (free)
2. Profile -> **Settings** -> **Access Tokens** -> **New token**
3. Token type: **Write**. Copy it - it starts with `hf_`

## Step 3: Publish the trained model (5 minutes, in Colab)

Run this in the Colab session where training completed (artifacts live in
your Drive). It stages exactly the files the browser runtime needs and
uploads them:

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

%cd /content/scamless
!git fetch origin && git reset --hard origin/main
!pip install -q huggingface_hub

import getpass, pathlib
token = getpass.getpass("Paste your HF WRITE token (hf_...): ")
assert token.startswith("hf_"), "token should start with hf_"

REPO = "RishavGupta01/scamless-model-v1"
model_dir = pathlib.Path("/content/drive/MyDrive/scamless/artifacts/model_v1")
assert (model_dir / "onnx" / "model_int8.onnx").exists(), (
    "exported artifact not found - run the export + eval cell first"
)

!python training/scripts/prepare_web_model.py --model-dir "{model_dir}"

from huggingface_hub import HfApi
api = HfApi(token=token)
api.create_repo(REPO, exist_ok=True, repo_type="model")
api.upload_folder(folder_path="publish/web-model", repo_id=REPO, repo_type="model")
print("LIVE: https://huggingface.co/" + REPO)
```

The staged bundle contains: `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`,
`onnx/model_int8.onnx`, and `thresholds.json` (the calibrated per-label
thresholds from the tuning pass).

## Step 4: Verify the full product (3 minutes)

1. Hard-reload `https://rishavgupta01.github.io/Scamless/` (Ctrl+Shift+R)
2. The status line must reach **"Model ready - running locally (WebGPU)"**
   (or WASM). The first load downloads ~118-470 MB one time; the meter
   shows real MB progress. After this it is cached and instant.
3. Click each sample chip. Expected per scan:
   - Risk meter animates to a 0-100 score, color-coded
   - Verdict band: SAFE / SUSPICIOUS / DANGEROUS
   - "Why" panel: model labels with fused confidence + detected signals
   - "Annotated message": urgency phrases, links, OTP codes and money
     amounts highlighted in place
   - "What to do now": category playbook
4. The engine tag under the buttons must read
   "full model - ran locally", not demo mode.

## Step 5: Chrome extension (3 minutes)

On your machine, from the repository root:

    python apps/extension/fetch-deps.py     # vendors the inference bundle

Then: `chrome://extensions` -> enable **Developer mode** -> **Load unpacked**
-> select the `apps/extension/` folder.

First scan downloads the model once into extension storage; afterwards it
works fully offline. Test flow:

1. Popup: paste a sample, Scan - verdict card with risk band appears
2. Select any text on a page, right-click -> "Check selected text for scams"
3. On Gmail/Outlook/WhatsApp Web: per-message "Check" buttons appear
4. Ctrl+Shift+S anywhere: scans the current selection
5. Toolbar badge shows the risk score, color-coded per tab

## Step 6: Chrome Web Store publication (optional, when ready)

Requirements: a one-time USD 5 developer registration
(`https://chrome.google.com/webstore/devconsole`), a privacy policy URL
(live at `/privacy.html` after step 1), and store icons + zipped package
(both generated):

    python training/scripts/generate_icons.py
    python training/scripts/make_store_zip.py

Upload `dist/scamless-extension-v<version>.zip`, copy the listing text from
`docs/store-listing.md` (includes the per-permission justifications reviewers
ask for), and submit. Review typically takes a few days. Until approved, the
extension is distributable as load-unpacked or via the repository.

## Step 7: Future model updates

1. Train: `python -m scamless.pipeline` (Colab, checkpoints on Drive)
2. The gate must pass: `macro_f1 >= macro_f1_min`,
   `false_positive_rate <= fp_rate_max`, `scam_union_f1 >= scam_union_f1_min`
   (values in `training/src/scamless/eval/eval_config.json`)
3. Re-run the step 3 cell - the upload overwrites the Hub revision
4. Frontends pick it up on next model load; users may need one hard reload

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Site shows "demo mode" after model upload | Repo id mismatch | `apps/web/js/config.js` MODEL_REPO must equal the uploaded repo |
| "Model ready" but downloads again every visit | Browser storage blocked | Allow site data / third-party cookies for the site |
| Extension scan says model fetch failed | Vendored bundle missing | Run `apps/extension/fetch-deps.py`, reload unpacked |
| Gate fails on re-run | Model quality regressed | Check per-label table in metrics.json; the failing categories are the data roadmap |
| Colab disconnects mid-training | Free tier session limit | Re-run the training cell: checkpoints on Drive auto-resume |

## Post-launch loop

1. Users mark wrong verdicts in the extension popup - stored locally,
   exportable as JSON
2. Exported corrections + new public corpora feed the next
   `python -m scamless.pipeline` run (the replay buffer mixes old data back
   so nothing is forgotten)
3. Gates verify quality did not regress before the update ships
