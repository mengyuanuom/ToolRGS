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
