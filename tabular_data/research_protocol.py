"""保持原始 CSV 的独立训练预算实验。"""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.model_selection import train_test_split


def training_budget(frame, fraction=1., seed=42):
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("训练预算比例必须在 (0,1] 内。")
    indices = np.arange(len(frame))
    if fraction < 1:
        indices, _ = train_test_split(indices, train_size=fraction, random_state=seed, stratify=frame.target)
        indices = np.sort(indices)
    selected = frame.iloc[indices].copy()
    counts = selected.target.astype(str).value_counts().sort_index()
    if set(counts.index) != set(frame.target.astype(str)) or counts.min() < 5:
        raise ValueError("受限预算必须覆盖所有类别，且每类至少 5 行；请增加训练预算。")
    return selected, {"requested_fraction_of_training_split": fraction, "seed": seed,
        "available_rows": len(frame), "used_rows": len(selected), "unused_rows": len(frame)-len(selected),
        "selected_source_row_positions": indices.tolist(),
        "selection_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "class_counts": {str(k): int(v) for k,v in counts.items()},
        "class_proportions": {str(k): float(v / len(selected)) for k,v in counts.items()},
        "method": "stratified subset of training split; validation/test unchanged",
        "unused_rows_used_elsewhere": False}
