# ToolRGS

Tool-oriented Referring Grasp Synthesis with a single configuration-driven
codebase for CROG, CROG-OFF, DROG, DROG-OFF, LGD, GGCNN-CLIP,
GR-ConvNet-CLIP, and DETRIS backbones. Grasp-Tools, VCoT/Grasp-Anything,
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

Available experiments:

- `config/grasp_tools/crog.yaml`
- `config/grasp_tools/crogoff.yaml`
- `config/grasp_tools/drog.yaml`
- `config/grasp_tools/drogoff.yaml`
- `config/grasp_tools/ggcnnclip.yaml`
- `config/grasp_tools/grconvnetclip.yaml`
- `config/grasp_tools/lgd.yaml`
- `config/vcot/crog.yaml`
- `config/ocid_vlg/crog.yaml`

Set `DATA.root_path`, `TRAIN.clip_pretrain`, and (for DROG variants)
`TRAIN.dino_pretrain` to local paths before training.

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

All registered ToolRGS models can use VCoT without code changes. Either copy
the data block above into a model config, or override the shared fields:

```bash
python train.py --config config/grasp_tools/drogoff.yaml --opts \
  DATA.dataset vcot \
  DATA.root_path /mnt/ssd0/mengyuan/data/grasp-anything \
  DATA.train_split train DATA.val_split unseen
```

## OCID-VLG data

OCID-VLG referring expressions are read directly from the downloaded dataset;
the large RGB, depth, and annotation files are not copied into this repository.
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

`LGD` is a ToolRGS dense-map port of Language-driven Grasp Detection. It keeps
the public cosine diffusion schedule, x0 quality-map denoising, language/image
conditioning, and contrastive alignment while exposing the shared segmentation,
quality, sine, cosine, and width contract. `TRAIN.lgd_sampling_steps` controls
the DDIM inference cost; use `1000` for the full training schedule or a smaller
value for faster comparison. The upstream LGD MIT notice is in
`model/lgd_LICENSE`. See the
[CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-driven_Grasp_Detection_CVPR_2024_paper.html)
and [official implementation](https://github.com/Fsoft-AIC/LGD).

## Environment

Use Python 3.9 and PyTorch 2.0.1. Install the dependencies from
`requirement.txt`. Pretrained CLIP and DINOv2 weights are not stored in Git.

## Acknowledgements

ToolRGS integrates ideas and code from CROG, DETRIS, DINOv2, and CRIS. Preserve
their citations and licenses when publishing derived results.
