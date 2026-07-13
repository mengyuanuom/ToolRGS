# ToolRGS

Tool-oriented Referring Grasp Synthesis with a single configuration-driven
codebase for CROG, CROG-OFF, DROG, DROG-OFF, LGD, GGCNN-CLIP,
GR-ConvNet-CLIP, and DETRIS backbones.

## Design

All architectures live below `model/` and are selected by `MODEL.architecture`
in YAML. Grasp-Tools is loaded through one `GraspToolDataset`, one collate
contract, and one training/evaluation engine.

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

Set `DATA.root_path`, `TRAIN.clip_pretrain`, and (for DROG variants)
`TRAIN.dino_pretrain` to local paths before training.

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
