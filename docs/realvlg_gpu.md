# RealVLG Native V3 LoRA on CUDA

The CUDA port provides the same Native V3 experiment profile as
`ToolRGSNPU/config/realvlg/native_v3_lora_r24_12l_bs128_e24.yaml`:

- DINOv2 and CLIP LoRA on all 12 transformer blocks;
- LoRA rank 24 and alpha 48;
- four native visual-language fusion stages;
- dense Offset V2 with the lightweight offset head;
- 24 epochs, base learning rate `1e-4`, and global batch size 128;
- the official RealVLG `seen` validation protocol and `gAcc` metric.

Place the dataset at `datasets/GraspNet_VLG` and pretrained weights at:

```text
pretrain/ViT-B-16.pt
pretrain/dinov2_vitb14_reg4_pretrain.pth
```

Train on eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --config config/realvlg/native_v3_lora_r24_12l_bs128_e24.yaml
```

The config marks `batch_size` as global, so eight workers receive 16 samples
each. It must train from scratch because its 12-layer rank-24 LoRA graph is not
compatible with older four-layer or rank-8 Native V3 checkpoints.

Evaluate one split on a single GPU:

```bash
python evaluate.py \
  --config config/realvlg/native_v3_lora_r24_12l_bs128_e24.yaml \
  --checkpoint exp/realvlg/drogoff_native_v3_lora_r24_12l_v2cap_realvlg_full_e24_8gpu_bs128/best_jindex_model.pth \
  --split seen \
  --opts batch_size_val 4
```

Use `--split similar` or `--split novel` for the other official partitions.
