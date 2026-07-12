import torch.nn as nn

from .layers import MultiTaskProjector   
from .layers import FiLMProjector            

def build_projector(cfg) -> nn.Module:
    """
    根据 config 里 MODEL.PROJECTOR.TYPE 构建不同的 head：
      - "dynamic" -> MultiTaskProjector (你现在的版本)
      - "film"    -> FiLMProjector (FiLM 版本)
    """
    proj_cfg = cfg.projector # 你按自己 config 结构改
    vis_dim  = cfg.vis_dim
    word_dim = cfg.word_dim

    proj_type   = proj_cfg.lower()
    if proj_type == "dynamic":
        projector = MultiTaskProjector(
            word_dim=word_dim,
            in_dim= vis_dim//2,
            kernel_size=3
        )
    elif proj_type == "film":
        projector = FiLMProjector(
            word_dim=word_dim,
            in_dim=vis_dim//2,
            kernel_size=3
        )
    else:
        raise ValueError(f"Unknown PROJECTOR.TYPE = {proj_cfg.TYPE}")

    print(f"[build_projector] Use projector type: {proj_type}, "
          f"word_dim={word_dim}, in_dim={vis_dim//2}, k={3}")
    return projector