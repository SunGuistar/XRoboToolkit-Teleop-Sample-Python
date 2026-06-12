"""Minimal BC policy: ResNet18 image encoder + state MLP → action."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BCPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 7,
        state_hidden: int = 128,
        fusion_hidden: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim + state_hidden, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        image: (B, 3, H, W)
        state: (B, state_dim)
        returns action: (B, action_dim)
        """
        img_feat = self.image_encoder(image)
        st_feat = self.state_mlp(state)
        fused = torch.cat([img_feat, st_feat], dim=-1)
        return self.head(fused)
