# ToolRGS

Tool-oriented Referring Grasp Synthesis with a single configuration-driven
codebase for CROG, CROG-OFF, DROG, DROG-OFF, ETRG-A, MapleGrasp, GraspMamba, LGD,
GGCNN-CLIP, GR-ConvNet-CLIP, and DETRIS backbones. Grasp-Tools, VCoT/Grasp-Anything,
and OCID-VLG data use the same model-facing batch contract.

## Design

All architectures live below `model/` and are selected by `MODEL.architecture`
in YAML. Datasets are selected by `DATA.dataset`; the registered adapters feed
one model-facing batch contract and one training/evaluation engine.

```text
ToolRGS/
├── model/
│   ├── crog.py
│   ├── crogoff.py
│   ├── drog.py
│   ├── drogoff.py
│   ├── ggcnnclip.py
│   ├── grconvnetclip.py
│   ├── graspmamba.py
│   ├── lgd.py
│   ├── segmenter.py
│   └── dinov2/
├── config/grasp_tools/
├── engine/engine.py
├── utils/dataset.py
├── train.py
└── evaluate.py
```

`DROGOFF` combines DROG's DINOv2 + CLIP-adapter fusion with a two-channel
normalized center-offset head. Offset supervision is generated from transformed
Grasp-Tools rectangle centers and weighted by a Gaussian `off_w` map.

## Configuration

Choose a model entirely from the experiment config:

```yaml
MODEL:
  architecture: drogoff
```

New experiments can use composable MMEngine-style `_base_` configs. Model,
dataset, schedule, and CUDA runtime settings live independently under
`configs/_base_/`:

```yaml
_base_:
  - ../_base_/datasets/ocid_vlg.yaml
  - ../_base_/models/etrg_r50.yaml
  - ../_base_/schedules/etrg_40e.yaml
  - ../_base_/runtime/cuda.yaml

TRAIN:
  exp_name: etrg_r50_ocid_vlg
  output_folder: exp/ocid_vlg
```

Preferred training entrypoint:

```bash
python tools/train.py --config configs/etrg/etrg_r50_ocid_vlg.yaml
```

`CUDAGraspRunner`, `CUDAAmpOptimWrapper`, registered schedulers, and runner
hooks own construction, CUDA AMP/backward, epoch scheduling, logging, and
checkpoints. The old `python train.py --config config/...` command remains
compatible.

The nine RGB model families are available for every dataset. ETRG-A is added
for OCID-VLG because it requires real aligned depth:

| Dataset config directory | Models |
| --- | --- |
| `config/grasp_tools/` | `crog`, `crogoff`, `drog`, `drogoff`, `maplegrasp`, `ggcnnclip`, `grconvnetclip`, `graspmamba`, `lgd` |
| `config/vcot/` | `crog`, `crogoff`, `drog`, `drogoff`, `maplegrasp`, `ggcnnclip`, `grconvnetclip`, `graspmamba`, `lgd` |
| `config/ocid_vlg/` | `crog`, `crogoff`, `drog`, `drogoff`, `etrg`, `maplegrasp`, `ggcnnclip`, `grconvnetclip`, `graspmamba`, `lgd` |

For example, `config/vcot/drogoff.yaml` and
`config/ocid_vlg/lgd.yaml` are directly runnable after setting data and weight
paths. DETRIS remains a referring-segmentation baseline and does not implement
the shared grasp-map output/loss contract, so it is not included in this matrix.

Set `DATA.root_path`, `TRAIN.clip_pretrain`, and (for DROG variants)
`TRAIN.dino_pretrain` to local paths before training.

## Embedded Grasp-Tools v2 data and augmentation

The complete Grasp-Tools source set is included in this repository: 107
annotated RGB images with JSON masks/grasps and 42 background images live under
`assets/grasp_tools/`. Generate the multi-object, multi-query v2 dataset from a
fresh clone with:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py
```

The default output is `datasets/grasp-tools/aug_graspall_v2`. It uses the
difficulty-1 starter curriculum: 6000 train, 500 validation, and 1000 test
scenes; two or three unique-category tools per scene; expanded category
vocabulary; shared language templates; 24 balanced rotation bins; and mild
appearance augmentation. Train the supplied DROG-OFF experiment with:

```bash
python train.py --config config/grasp_tools/drogoff_v2.yaml
```

See `docs/grasp_tools_v2.md` for the smoke test, output schema, and
full generation options.

## VCoT / Grasp-Anything data

The repository includes only the official VCoT CSV split metadata. Keep the
large image, `.pt`, and `.npy` files outside Git in this layout:

```text
/path/to/grasp-anything/
├── image/<scene_id>.jpg
├── positive_grasp/<grasp_id>.pt
└── mask/<grasp_id>.npy
```

Set the dataset in YAML:

```yaml
DATA:
  dataset: vcot
  root_path: /path/to/grasp-anything
  split_root: ./split/vcot
  train_split: train
  val_split: unseen       # or seen
  prompt_template: "Grasp the {object_name}"
```

The adapter reads `.pt` grasps as
`[score, x, y, width, height, theta_degrees]`, discards the score for geometry,
reorders the quadrilateral for ToolRGS's width/angle convention, and generates
grasp maps after letterboxing. Original-coordinate grasp
targets are retained for Jacquard evaluation. Files are loaded lazily per
sample; the dataset does not preload the full annotation corpus.

Inspect the same sample you verified previously:

```bash
python tools/inspect_vcot_sample.py \
  --dataset-root /mnt/ssd0/mengyuan/data/grasp-anything \
  --csv split/vcot/train.csv --row 2
```

All nine grasp-aware ToolRGS models can use VCoT without code changes. Use the
matching file under `config/vcot/`, for example:

```bash
python train.py --config config/vcot/drogoff.yaml --opts \
  DATA.root_path /mnt/ssd0/mengyuan/data/grasp-anything
```

### Recommended VCoT profile for two RTX 3090 GPUs

VCoT YAML batch sizes and worker counts are per distributed process (per GPU).
The checked-in profile targets two 24 GB RTX 3090 cards:

| Model | Input | Train batch/GPU | Global batch | Epochs | LR milestones |
| --- | ---: | ---: | ---: | ---: | --- |
| CROG / CROG-OFF | 416 | 8 | 16 | 70 | 55, 65 |
| MapleGrasp | 416 | 8 | 16 | 70 | 55, 65 |
| DROG / DROG-OFF | 448 | 8 | 16 | 65 | 35, 55 |
| GGCNN-CLIP | 416 | 32 | 64 | 50 | 35 |
| GRConvNet-CLIP | 416 | 32 | 64 | 80 | 70 |
| GraspMamba | 416 | 8 | 16 | 50 | 35, 45 |
| LGD | 224 | 16 | 32 | 100 | 70, 90 |

Most VCoT profiles use eight training workers and four validation workers per
process. The heavier DROG-OFF profile uses four and two respectively, avoiding
CPU/RAM and shared-memory pressure when two GPU processes run together. Start
a two-GPU run with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config config/vcot/graspmamba.yaml --opts \
  DATA.root_path /mnt/ssd0/mengyuan/data/grasp-anything
```

If a heavy model runs out of memory, reduce both per-GPU batches without
editing YAML, for example `TRAIN.batch_size 4 TRAIN.batch_size_val 4`.

## OCID-VLG data

OCID-VLG referring expressions are read directly from the downloaded dataset;
the large RGB, depth, and annotation files are not copied into this repository.
Download the complete `OCID-VLG.zip` from the
[official OCID-VLG repository](https://github.com/gtziafas/OCID-VLG) or its
[official Google Drive file](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link).

For the standard server checkout at `/mnt/ssd0/mengyuan/ToolRGS`, put the data
next to the repository at `/mnt/ssd0/mengyuan/data/OCID-VLG`. The checked-in
`datasets` symlink points to `../data`, so every OCID-VLG YAML can keep
`DATA.root_path: ./datasets/OCID-VLG`:

```bash
python -m pip install gdown
mkdir -p /mnt/ssd0/mengyuan/data
cd /mnt/ssd0/mengyuan/data
gdown --fuzzy \
  'https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link' \
  -O OCID-VLG.zip
unzip OCID-VLG.zip
```

After extraction, use the directory that directly contains `refer/` as the
dataset root. If the archive creates an extra nested directory, either move it
or override `DATA.root_path` with that directory's absolute path.
The expected layout is:

```text
/path/to/OCID-VLG/
├── refer/multiple/
│   ├── train_expressions.json
│   ├── val_expressions.json
│   └── test_expressions.json
└── <sequence>/
    ├── rgb/<image_name>
    ├── depth/<image_name>
    └── seg_mask_instances_combi/<image_name>
```

Use the supplied experiment or select the dataset from another model config:

```yaml
DATA:
  dataset: OCID-VLG
  root_path: /path/to/OCID-VLG
  version: multiple
  with_depth: true
  train_split: train
  val_split: val
```

The adapter keeps original-coordinate grasp rectangles for Jacquard evaluation,
then transforms the rectangle corners before generating input-resolution grasp
maps. This avoids the fixed-416 map misalignment in the legacy loader. It also
supports the center-offset supervision required by CROG-OFF and DROG-OFF.

Inspect one expression before training:

```bash
python tools/inspect_ocid_vlg_sample.py \
  --dataset-root /path/to/OCID-VLG \
  --version multiple --split train --index 0
```

On the standard server layout the exact check is:

```bash
cd /mnt/ssd0/mengyuan/ToolRGS
readlink -f datasets
test -f datasets/OCID-VLG/refer/multiple/train_expressions.json
python tools/inspect_ocid_vlg_sample.py \
  --dataset-root datasets/OCID-VLG --version multiple --split train --index 0
```

Train any supported grasp model with its OCID-VLG config, for example:

```bash
python train.py --config config/ocid_vlg/drog.yaml --opts \
  DATA.root_path /path/to/OCID-VLG
```

## ETRG-A

The `etrg` registry entry integrates the official parameter-efficient ETRG-A
RGB-D architecture. Unlike the RGB-only models, it routes the aligned
OCID-VLG depth map through a ResNet-18 encoder and fuses depth, visual, and
language tokens inside the frozen CLIP backbone adapters.

Run the R50 profile on two GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config config/ocid_vlg/etrg.yaml --opts \
  DATA.root_path /path/to/OCID-VLG \
  TRAIN.clip_pretrain pretrain/RN50.pt
```

The R101 alternative is `config/ocid_vlg/etrg_r101.yaml`. ETRG requires real
aligned depth, so VCoT and Grasp-Tools configs are intentionally not provided.
See [docs/etrg.md](docs/etrg.md) for weights, offline setup, and compatibility
details.

## MapleGrasp

`model/maplegrasp.py` ports the official CROG-based MapleGrasp mask-guided
projector into ToolRGS. A detached predicted segmentation mask gates the four
grasp feature branches before the dynamic language-conditioned convolution.
Unlike the released reference implementation, the ToolRGS port avoids a fixed
104x104 feature size and never consumes a ground-truth object mask at
evaluation time.

The directly runnable configuration uses joint segmentation/grasp training:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/ocid_vlg/maplegrasp.yaml --opts \
  DATA.root_path /path/to/OCID-VLG
```

For the paper-style two-stage schedule, first train only segmentation and then
initialize the grasp stage from the best-IoU checkpoint:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/ocid_vlg/maplegrasp.yaml --opts \
  DATA.root_path /path/to/OCID-VLG \
  TRAIN.maple_stage segmentation \
  TRAIN.exp_name maplegrasp_ocid_vlg_stage1

torchrun --nproc_per_node=2 train.py \
  --config config/ocid_vlg/maplegrasp.yaml --opts \
  DATA.root_path /path/to/OCID-VLG \
  TRAIN.maple_stage grasp \
  TRAIN.weight exp/ocid_vlg/maplegrasp_ocid_vlg_stage1/best_iou_model.pth \
  TRAIN.exp_name maplegrasp_ocid_vlg_stage2
```

`TRAIN.weight` initializes model parameters only; `TRAIN.resume` additionally
restores epoch, optimizer, and scheduler state. Configured local files are
checked before model construction, so missing CLIP or checkpoint paths now fail
with the resolved path and a direct remediation message.

This port follows the [MapleGrasp paper](https://arxiv.org/abs/2506.06535) and
[official implementation](https://github.com/vineet2104/MapleGrasp). Report
ToolRGS results separately from the paper until the same split and two-stage
schedule have been reproduced.

## Training

Single GPU:

```bash
python train.py --config config/grasp_tools/drogoff.yaml
```

Distributed:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/grasp_tools/drogoff.yaml
```

Evaluation:

```bash
python evaluate.py \
  --config config/grasp_tools/drogoff.yaml \
  --checkpoint exp/grasp_tools/drogoff_grasp_tools/best_jindex_model.pth
```

## Output contract

Grasp-aware models return segmentation, quality, sine, cosine, and width maps.
Offset variants append a `(dx, dy)` map normalized by `DATA.offset_r`.
GGCNN-CLIP and GR-ConvNet-CLIP are grasp-only baselines, so their quality map
also occupies the segmentation slot required by the shared engine.

For DROG-OFF evaluation, `TEST.offset_resample_geometry: true` first refines the
quality-peak center and then bilinearly re-reads angle and width at that refined
center. Set it to `false` to reproduce the legacy center-only decoder in an
ablation; this switch changes only decoding and does not require retraining.

`LGD` is a ToolRGS dense-map port of Language-driven Grasp Detection. It keeps
the public cosine diffusion schedule, x0 quality-map denoising, language/image
conditioning, and contrastive alignment while exposing the shared segmentation,
quality, sine, cosine, and width contract. `TRAIN.lgd_sampling_steps` controls
the DDIM inference cost; use `1000` for the full training schedule or a smaller
value for faster comparison. The upstream LGD MIT notice is in
`model/lgd_LICENSE`. See the
[CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-driven_Grasp_Detection_CVPR_2024_paper.html)
and [official implementation](https://github.com/Fsoft-AIC/LGD).

`GraspMamba` is a ToolRGS paper reimplementation, not the unreleased official
training code. It follows the paper's four-stage MambaVision backbone, frozen
CLIP text encoder, per-stage visual-language fusion, and recursive top-down
feature aggregation. The adapter adds an instance-segmentation head and emits
the shared dense grasp maps required by the ToolRGS engine. VCoT/Grasp-Anything
is the paper-aligned training dataset; the Grasp-Tools and OCID-VLG configs are
cross-dataset compatibility experiments rather than paper-reported settings.
See the [GraspMamba paper](https://arxiv.org/abs/2409.14403) and the
[official MambaVision backbone](https://github.com/NVlabs/MambaVision).

Run the paper-aligned experiment with:

```bash
python train.py --config config/vcot/graspmamba.yaml --opts \
  DATA.root_path /mnt/ssd0/mengyuan/data/grasp-anything
```

## Environment

Use Python 3.9 and PyTorch 2.0.1. Install the dependencies from
`requirement.txt`. Pretrained CLIP and DINOv2 weights are not stored in Git.

GraspMamba has an optional CUDA extension and must be installed after the
CUDA-matched PyTorch build:

```bash
pip install "mamba-ssm==2.2.4" --no-build-isolation
pip install -r requirement-mamba.txt
python tools/check_graspmamba_env.py --clip pretrain/RN50.pt
```

`--no-build-isolation` is important because the CUDA extension must compile
against the PyTorch already installed in the active environment.

The configured official MambaVision checkpoint is downloaded automatically if
it is missing and the server has network access. Otherwise download it once and
set `TRAIN.mamba_pretrain` to the local file. MambaVision code uses NVIDIA's
non-commercial source license and its pretrained weights use CC-BY-NC-SA-4.0;
check those terms before redistribution or commercial use.

## Real-world demo and robot sender

ToolRGS includes a configuration-driven PyQt demo ported from the local server
CROG deployment. It supports all nine ToolRGS grasp architectures, OpenCV/video,
RealSense, GStreamer shared memory, optional MMDetection and Whisper, and the
legacy Kinova TCP command format. Start in dry-run mode:

```bash
cp config/deployment/lab.example.yaml config/deployment/lab.yaml
python tools/check_deployment.py --config config/deployment/lab.yaml \
  --probe-camera --build-model
python deploy_gui.py --config config/deployment/lab.yaml
```

See [docs/real_world_deployment.md](docs/real_world_deployment.md) before
enabling robot output. The repository contains the sender but not the external
Kinova receiver/controller or its calibration, so a clone alone cannot safely
move the physical robot.

## Component architecture

ToolRGS now has MMDetection-style registries for models, datasets, transforms,
losses, metrics, postprocessors, loops, hooks, cameras, robot clients,
detectors, and audio inputs.
Existing training configs and builders remain compatible while new components
can be selected by `type` without extending central `if/elif` factories.

```bash
python tools/list_components.py
```

Dense model tuples can be normalized into named `GraspOutput`, `GraspTargets`,
and `GraspModelResult` structures. The main paths now use registered
`GraspTrainLoop` and `GraspValLoop` components; validation and deployment share
the `DenseGraspPostProcessor` decoding contract. See
[docs/component_architecture.md](docs/component_architecture.md) for extension
examples and the compatibility plan.

## Acknowledgements

ToolRGS integrates ideas and code from CROG, MapleGrasp, DETRIS, DINOv2, and CRIS. Preserve
their citations and licenses when publishing derived results.
