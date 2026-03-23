# Advanced Training Stability Metrics — Walkthrough

Successfully updated `finetune_donut.ipynb` with three new metrics and label-aware data loading. Changes pushed to `main`.

## Phase 1 — `DonutDataset` (Cell 5)

`__getitem__` now reads the `label` column from the HuggingFace dataset (set by `generate_metadata.py`) and returns it as `label_type` alongside `pixel_values` and `labels`. Examples: `"notes"`, `"coord table"`, `"azimuth"`.

## Phase 2 — `DonutModelPLModule` (Cell 6)

### New State
| Field | Purpose |
|---|---|
| `self.best_cer` | Lowest CER ever reached; updated each epoch |
| `self.epoch_cer_history` | All past epoch means; used for rolling window |
| `self.validation_step_outputs` | Temp list of `{cer, label}` dicts; cleared each epoch |

### New Metrics Logged

| Metric | Logged When | Description |
|---|---|---|
| `val_cer` | Per batch | Batch-level average CER (%) |
| `best_cer_so_far` | End of epoch | Best single-epoch CER reached so far |
| `rolling_3_cer` | End of epoch | Mean CER of last ≤3 epochs (noise-resistant) |
| `cumulative_avg_cer` | End of epoch | Mean of all epoch CERs |
| `val_cer_<label>` | End of epoch | Per-category CER (e.g. `val_cer_coord_table`) |

### Example epoch output
```text
  [Epoch 3] epoch_cer=8.50%  rolling_3=10.33%  best=8.50%  cumulative=11.17%
  Per-label: azimuth=5.1%  coord table=9.3%  notes=11.2%
```

## Verification Results
- [x] Script reported `SUCCESS: Both cells updated.`
- [x] JSON validity check: `PASSED`
- [x] Committed and pushed to GitHub (`0d07eab`)

> [!NOTE]
> `label_type` is returned as a plain string from the `DataLoader`. The `on_validation_epoch_end` hook uses `defaultdict` to group CER values by label string before averaging.
