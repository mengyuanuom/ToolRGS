import torch.nn as nn


class TextVisualFusionFiLM(nn.Module):
    """Text-conditioned channel modulation shared by CLIP grasp heads."""

    def __init__(self, vis_dim: int, text_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Linear(hidden_dim, vis_dim)
        self.beta = nn.Linear(hidden_dim, vis_dim)

    def forward(self, feat, e_txt):
        batch_size, channels, _, _ = feat.shape
        text = self.mlp(e_txt)
        gamma = self.gamma(text).view(batch_size, channels, 1, 1)
        beta = self.beta(text).view(batch_size, channels, 1, 1)
        return feat * (1.0 + gamma) + beta
