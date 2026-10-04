"""Val-only temperature scaling audit for already locked F01 predictions."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from eval import compute_metrics

predictions = root / "submissions" / "2A202602759_NguyenMinhHien" / "predictions"


def scaled_probabilities(probabilities, temperature):
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
    result = np.exp(log_probabilities)
    return result / result.sum(axis=1, keepdims=True)


rows = []
for seed in (0, 1, 2):
    frame = pd.read_csv(predictions / f"F01_seed{seed}_val.csv")
    y = frame.y_true.to_numpy(dtype=int)
    p = frame[[f"p{i}" for i in range(9)]].to_numpy(dtype=float)
    before = compute_metrics(y, p.argmax(axis=1), p)

    def nll(log_t):
        q = scaled_probabilities(p, math.exp(log_t))
        return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean())

    a, b = math.log(0.2), math.log(5.0)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(80):
        if nll(c) < nll(d):
            b, d = d, c
            c = b - ratio * (b - a)
        else:
            a, c = c, d
            d = a + ratio * (b - a)
    temperature = math.exp((a + b) / 2)
    q = scaled_probabilities(p, temperature)
    after = compute_metrics(y, q.argmax(axis=1), q)
    rows.append({"seed": seed, "fit_split": "val", "n": len(y),
                 "temperature": temperature, "ece_before": before["ece"],
                 "ece_after": after["ece"], "nll_before": before["nll"],
                 "nll_after": after["nll"],
                 "macro_f1_before": before["macro_f1"],
                 "macro_f1_after": after["macro_f1"]})

out = root / "submissions" / "2A202602759_NguyenMinhHien" / "temperature_val.json"
out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
for row in rows:
    print(row)
