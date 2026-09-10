// Scamless - deployment configuration.
// Point MODEL_REPO at a Hugging Face repo containing the exported artifact:
//   config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json,
//   vocab.txt, onnx/model_int8.onnx, thresholds.json
// See docs/model-hosting.md for the one-command upload.

export const CONFIG = {
  MODEL_REPO: "Rishavgupta/scamless-model-v1",
  MAX_LEN: 256,
  // risk bands (0-100), per spec section 9
  BANDS: { suspicious: 40, dangerous: 70 },
  // below this probability a label is shown as a weak signal in the why panel
  WEAK_SIGNAL: 0.35,
};
