# Grasp-Tools V4 Dense generation

V4 Dense is generated again from the 107 original object images and the reviewed
polygon masks and grasp rectangles in `assets/grasp_tools/graspall_v3`. It does
not reuse any V3 composite scene.

## Dataset contract

- 12,000 train, 1,000 validation, and 2,000 test scenes by default.
- Every scene contains a randomly requested 10--12 successfully placed objects.
- Canonical categories are unique inside one scene.
- Every placed object creates exactly one category-only language query. No color,
  left/right, or relational phrase is used.
- Nominal object scales are 0.3, 0.4, 0.5, and 0.6. A failed placement is
  retried at progressively smaller scales, never below 0.3. If it still cannot
  be placed, another category is tried. An underfilled scene is discarded.
- Rotation uses 24 bins over 360 degrees (15 degrees per bin) with continuous
  plus/minus 7.5-degree jitter. RGB, mask, and grasp rectangles share the same
  affine transform.
- Category, source-instance, scale, and angle-bin counters advance only after a
  complete scene succeeds. Counters are independent for train/validation/test.

The downstream grasp representation remains unchanged: angles are normalized to
the 180-degree gripper symmetry convention and trained through `sin(2 theta)` /
`cos(2 theta)` by the dataset loader.

## Generate

From the repository root:

```bash
bash tools/generate_grasp_tools_v4_dense.sh --overwrite
```

The default output is
`datasets/grasp-tools/aug_graspall_v4_dense_15k`. Override it and the scene counts
through environment variables when making a pilot:

```bash
OUT_DIR=/tmp/grasp_tools_v4_pilot \
TRAIN_SCENES=400 VAL_SCENES=50 TEST_SCENES=50 \
bash tools/generate_grasp_tools_v4_dense.sh --overwrite
```

The generator automatically runs `validate_v4_dense.py` after generation. The
validator rejects underfilled scenes, repeated categories, missing per-object
queries, scales outside 0.3--0.6, rotations outside their assigned bin, unequal
placement/query distributions, and category or successful-angle imbalance.
