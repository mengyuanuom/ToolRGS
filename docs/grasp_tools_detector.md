# Grasp-Tools Faster R-CNN detector

This pipeline trains the GUI object detector with MMDetection rather than the
old unrelated `epoch_48_13.pth` checkpoint.

## Dataset contract

The detector uses the same rendered V2 scenes as language grasp training:

```text
datasets/grasp-tools/aug_graspall_v2/
├── train/
├── val/
└── test/
```

Every scene JSON contains an `objects` list and multiple language `queries`.
The COCO converter uses each entry in `objects` exactly once. It does not count
the same object again for each query.

The 22 detection classes are:

```text
tape measure, T-hex key, L-hex key, marker, wrench, pliers, mallet,
screwdriver, clamps, spool, sponge, clip, crimp tool, screw, tape, box,
nut, ruler, file, stapler, scissors, cable
```

T-hex key and L-hex key are separate classes. The class order is frozen in both
the converter and MMDetection config and must be preserved by deployment.

## Install MMDetection

Install a CUDA PyTorch build first, then use the repository detector pins:

```bash
pip install -U openmim
mim install "mmengine==0.10.7" "mmcv==2.1.0"
pip install -r requirement-detector.txt
```

Verify:

```bash
python -c "import torch, mmcv, mmengine, mmdet; print(torch.__version__, mmcv.__version__, mmengine.__version__, mmdet.__version__)"
```

## Convert to COCO and inspect

```bash
python tools/dataset_converters/grasp_tools/to_coco_detection.py \
  --dataset-root datasets/grasp-tools/aug_graspall_v2
```

It creates:

```text
annotations/grasp_tools_instances_train.json
annotations/grasp_tools_instances_val.json
annotations/grasp_tools_instances_test.json
```

The command fails if a paired JSON is missing, a polygon/bbox is invalid, an
unknown class appears, or any split lacks one of the 22 classes.

## Train on two RTX 3090 GPUs

The recommended config is Faster R-CNN R50-FPN, AMP, two images per GPU,
24 epochs, validation every two epochs and best checkpoint selection by COCO
bbox mAP. Start it in tmux:

```bash
cd /mnt/ssd0/mengyuan/ToolRGS
tmux new -s grasp_tools_detector
mkdir -p work_dirs/faster_rcnn_r50_fpn_grasp_tools_v2_24e
bash tools/train_grasp_tools_detector_2gpu.sh \
  2>&1 | tee work_dirs/faster_rcnn_r50_fpn_grasp_tools_v2_24e/train_console.log
```

Detach with `Ctrl-b d`; return with:

```bash
tmux attach -t grasp_tools_detector
```

Resume from the MMEngine `last_checkpoint` pointer:

```bash
TOOLRGS_DETECTOR_RESUME=1 bash tools/train_grasp_tools_detector_2gpu.sh
```

For a dataset stored elsewhere:

```bash
TOOLRGS_GRASP_TOOLS_ROOT=/absolute/path/to/aug_graspall_v2 \
  bash tools/train_grasp_tools_detector_2gpu.sh
```

## Formal test

Select the checkpoint recorded as `best_coco_bbox_mAP_epoch_*.pth`, then run:

```bash
export TOOLRGS_GRASP_TOOLS_ROOT=/mnt/ssd0/mengyuan/ToolRGS/datasets/grasp-tools/aug_graspall_v2
CUDA_VISIBLE_DEVICES=0 python tools/test_detector.py \
  configs/detection/faster_rcnn_r50_fpn_grasp_tools_v2_24e.py \
  work_dirs/faster_rcnn_r50_fpn_grasp_tools_v2_24e/best_coco_bbox_mAP_epoch_XX.pth
```

Record bbox mAP, mAP50, mAP75, small/medium/large AP and the classwise AP table.

## Deployment hand-off

The best validation checkpoint has now been published and wired into the GUI:

1. local path: `weights/faster_rcnn_r50_fpn_grasp_tools_v2_best.pth`;
2. Release asset: `faster_rcnn_r50_fpn_grasp_tools_v2_epoch24_best_map.pth`;
3. SHA-256: `76a4a09164f5de1a410957f4439f801328cf543f28f34fa6fee24a7f7eb49e74`;
4. validation bbox mAP/mAP50/mAP75: `0.972 / 0.999 / 0.996`;
5. run `python tools/check_deployment.py --build-detector` to download, verify,
   and load it before real RealSense/GI experiments.
