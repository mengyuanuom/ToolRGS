# GraspTools raster/angle scoring correction

The old rectangle evaluator passed x,y to a row,column polygon function with
shape H,W. This discarded image-right pixels. It also used a non-periodic
angle distance. Version `raster_xy_periodic180_v2` fixes both. Integer raster
IoU, strict IoU > threshold, and angle <= threshold are retained.

The evaluation loop now passes actual original image H,W and stores it in
new prediction caches. Old cached IoUs must not be reused: cache scoring
rebuilds them from the saved rectangles and targets, requiring an explicit
verified image shape when old archives lack bounds:

```bash
python evaluate.py --score-cache path/to/old.npz --image-shape 720 1280 --msr-output corrected.tsv
```

No model inference is needed. This command does not overwrite the old archive.
New caches carry a geometry version and can subsequently be threshold-scored
without rebuilding matches. Non-finite geometry is rejected.

For the unified GraspTools V3 inventory (1280x720 original images):

```bash
python tools/rescore_grasptools_geometry.py --root exp/grasp_tools \
  --output /path/to/new/results --image-shape 720 1280 --workers 3
```

The batch utility rejects test caches outside the audited original-coordinate
300-pixel width / 20-pixel short-side contract. It preserves predicted boxes,
model-specific sigmoid/clamp decoding, and checkpoint selection. It records
source/target/prediction hashes, old/new scores, thresholds, and evaluator
commit, and writes separate corrected prediction archives. It does not convert
old 100-pixel-trained checkpoints into 300-pixel-trained models.

This is a fixed-checkpoint rescore, not a claim that old validation-selected
checkpoints are optimal under corrected scoring. Model ranking and manuscript
tables must be rebuilt from the audited comparable caches. Different datasets
or image dimensions need their own verified bounds and protocol review.

## Completed test rescore

All **14 available test caches** completed. Each contains 5,029 queries with
identical target arrays and target offsets (SHA-256 prefix `58bdfbff19`).
The complete provenance, checkpoint paths, source hashes and 12-cell grids are
in `experiments/grasptools_geometry_20260912/results.json`. Scoring code commit
is `268d043`; the subsequent documentation/test-only commit is `2d2b292`.

| Checkpoint variant | J@1 (%) | J@5 (%) | mSR@1 (%) | mSR@5 (%) |
|---|---:|---:|---:|---:|
| DROG-OFF V1 | 99.13 | 99.22 | 87.80 | 92.46 |
| GraspMamba | 99.36 | 99.40 | 87.25 | 90.38 |
| DROG-OFF LoRA sigmoid | 97.73 | 97.85 | 85.45 | 89.62 |
| DROG-OFF V2 original-scale | 96.72 | 97.12 | 81.22 | 85.27 |
| CROG sigmoid | 98.11 | 98.15 | 78.62 | 82.77 |
| DROG sigmoid, incomplete epoch-24 run | 98.31 | 98.37 | 77.92 | 84.75 |
| CROG preceding non-sigmoid-width variant | 97.67 | 97.75 | 76.19 | 80.43 |
| MapleGrasp Stage 2, previously evaluated checkpoint | 95.76 | 95.96 | 68.95 | 73.04 |
| ETRG fixed, epoch 32 | 89.20 | 89.60 | 61.95 | 65.92 |
| LGD | 71.37 | 72.28 | 43.29 | 46.02 |
| GRConvNet-CLIP | 49.49 | 50.65 | 28.08 | 31.72 |
| GGCNN-CLIP fixed, epoch 36 | 45.91 | 50.07 | 26.29 | 32.16 |
| ETRG preceding implementation | 14.14 | 14.99 | 2.90 | 3.14 |
| GGCNN-CLIP preceding GRConv-FiLM implementation | 8.17 | 11.14 | 2.16 | 3.10 |

Rows are ordered by mSR@1, not J@1. GraspMamba leads J@1 in this inventory;
DROG-OFF V1 leads mSR@1. Versions are kept separate: the last two rows are not
the repaired ETRG/GGCNN checkpoints. The MapleGrasp row is not the newer
`sigmoid_consistent` experiment. These are not equal-training-budget ablations.

Image width/height is 1280/720; grasp width cap remains 300 and short side 20.
The original targets include 678 rectangles wider than 300 out of 383,822
target entries; they retain the documented 300 cap, not a 100 cap. No predicted
rectangle was changed. New archives contain corrected matches only. Thresholds
remain IoU strictly > 0.25 for J and periodic angle <= 30 degrees; mSR uses the
12 pairs documented above. This remains annotation-based planar grasp scoring,
not physical robot success.

### Scope limitations

This replaces the previously reported cached test results. It does not cover
uncached experimental checkpoints (for example DROG-LR and newer MapleGrasp
sigmoid-consistent), V4, or the other datasets. Their results must not be
invented from these caches. Selection of the best epoch under the corrected
validation metric is a separate audit. The existing paper/GUI score labels
have not been edited by this rescore.
