# ToolRGS component architecture

ToolRGS is migrating incrementally to an MMDetection-style architecture. The
first stage introduces shared registries and named model results without
breaking existing experiment YAML, checkpoints, imports, or training commands.

## Registries

The global registries live in `toolrgs/registry.py`:

```text
MODELS          DATASETS        TRANSFORMS
METRICS         POSTPROCESSORS  CAMERAS
ROBOT_CLIENTS   DETECTORS       AUDIO_INPUTS
```

Names are case-insensitive and normalize spaces/hyphens to underscores. Both
decorator and direct registration are supported:

```python
from toolrgs.registry import CAMERAS

@CAMERAS.register_module(name="my_camera", aliases=("lab-camera",))
def build_my_camera(camera_cfg, repo_root):
    return MyCamera(camera_cfg["device"])
```

The deployment config can then select it without changing a central builder:

```yaml
camera:
  type: my_camera
  device: 0
```

Run the component inventory in a configured training environment:

```bash
python tools/list_components.py
python tools/list_components.py --group models
```

## Model output contract

New code should use the structures in `toolrgs/structures`:

```python
GraspOutput(
    segmentation=seg,
    quality=quality,
    sine=sine,
    cosine=cosine,
    width=width,
    offset=offset,  # optional
)
```

`GraspModelResult` groups predictions, targets, the scalar training loss, and
named loss terms. `GraspModelResult.from_legacy(...)` accepts all historical
ToolRGS return layouts:

```text
(seg, quality, sine, cosine, width)
(seg, quality, sine, cosine, width, offset)
(predictions, targets)
(predictions, targets, total_loss, loss_dict)
```

Deployment inference already normalizes legacy model results through this
structure. `LegacyOutputAdapter` lets new loops consume an existing model as a
structured-output module, while the current training engine continues receiving
its historical tuples.

## Compatibility layer

- `model.MODEL_REGISTRY` remains available as a read-only view of `MODELS`.
- `utils.data_builder.DATASET_REGISTRY` remains available as a read-only view of
  `DATASETS`.
- `build_model(cfg)` still returns `(model, optimizer_parameter_groups)`.
- `build_dataset(cfg, split, with_offset)` keeps its existing signature.
- Dataset-specific optional arguments are signature-filtered; custom registered
  datasets can receive additional values through `DATA.dataset_args`.
- Deployment YAML accepts the new `type` field and still understands the old
  camera `backend` field.

## Migration sequence

This commit deliberately does not rewrite the training engine. The safe next
stages are:

1. register transforms, postprocessors, losses, and metrics;
2. move validation into `ValLoop` plus grasp/segmentation metric components;
3. move optimization into `TrainLoop` and hooks;
4. convert each model to `BaseGraspModel` and remove tuple adapters;
5. introduce composable `_base_` experiment configs after all legacy configs
   have parity tests.

At every stage the old CLI remains a compatibility entry until equivalent
training/evaluation results have been checked.
