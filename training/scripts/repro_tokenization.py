"""Local reproduction of the Colab tokenization death.

Rebuilds the exact dataset Colab built (224k rows) and runs the exact
tokenizer call build_dataset makes. If this crashes locally, we find the
poison row. If it passes, the data is clean and Colab died of RAM.
"""

import sys
import time

sys.path.insert(0, ".")

from transformers import AutoTokenizer

from scamless.data.build import collect_messages, dedupe

print("collecting...", flush=True)
messages, extra_urls = collect_messages()
print(f"collected {len(messages)} raw records", flush=True)

records = dedupe(messages)
print(f"after dedupe+clean: {len(records)}", flush=True)

import pandas as pd

df = pd.DataFrame(records)
texts = df["text"].tolist()
print(f"tokenizing {len(texts)} texts with the exact trainer call...", flush=True)

tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

t0 = time.time()
CHUNK = 5000
enc_ids = []
for i in range(0, len(texts), CHUNK):
    chunk = texts[i : i + CHUNK]
    enc = tok(chunk, truncation=True, max_length=256, add_special_tokens=True)
    enc_ids.extend(enc["input_ids"])
    if (i // CHUNK) % 10 == 0:
        print(f"  tokenized {i + len(chunk)}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)

print(f"tokenization completed without crash: {len(enc_ids)} rows in {time.time() - t0:.0f}s", flush=True)
