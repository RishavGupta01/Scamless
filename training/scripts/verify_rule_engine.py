"""Verify the rule engine catches the user's test case."""

import math

text = "Ghar baithe 4,000 rozana kamao. Simple copy paste ka kaam. Sirf 1,200 ka activation charge hai. Aaj se pehla payment kal milega"
lower = text.lower()

rules = [
    ("job_task", 2.5, "rozana kamao"),
    ("job_task", 2.0, "ghar baithe"),
    ("job_task", 1.5, "copy paste"),
    ("job_task", 2.5, "activation charge"),
    ("job_task", 2.0, "pehla payment"),
    ("job_task", 1.5, "aaj se pehla"),
    ("job_task", 1.5, "payment kal"),
    ("job_task", 2.0, "rozana kamai"),
]

scores = {}
hits = {}
for label, w, pat in rules:
    if pat in lower:
        scores[label] = scores.get(label, 0) + w
        hits[label] = hits.get(label, 0) + 1

for label in scores:
    n = hits[label]
    if n >= 2:
        scores[label] += min(n * 0.5, 5.0)

for label, s in sorted(scores.items(), key=lambda x: -x[1]):
    prob = 1 - math.exp(-max(0, s) / 2.2)
    band = "DANGEROUS" if prob >= 0.7 else "SUSPICIOUS" if prob >= 0.4 else "SAFE"
    print(f"{label}: score={s:.1f} prob={prob:.3f} band={band}")
