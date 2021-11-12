from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch import nn


class VisualFeatureCNNExtractor(nn.Module):

    def __init__(self, img_dim: int = 64, device='cuda'):
        super().__init__()

        self.device = device
        self.img_dim = img_dim

        model = []
        """ Convolutional part """
        in_feats = 3
        out_feats = in_feats
        for _ in range(2):
            model += [
                nn.Conv2d(
                    in_channels=in_feats,
                    out_channels=out_feats,
                    kernel_size=5,
                    padding=2,
                    stride=1),
                nn.BatchNorm2d(out_feats),
                nn.ReLU()
            ]
            model += [nn.MaxPool2d(kernel_size=2, stride=2)]
            in_feats = out_feats
            out_feats *= 1

        self.model = nn.Sequential(*model)

        in_feats_head = int(in_feats * (self.img_dim / 2**2)**2)
        self.extractor_head = nn.Sequential(
            nn.Linear(in_features=in_feats_head, out_features=in_feats_head),
            nn.ReLU(),
            nn.Linear(in_features=in_feats_head, out_features=in_feats_head),
            nn.ReLU(),
            nn.Linear(in_features=in_feats_head, out_features=128),
        ).to(self.device)

        self.to(self.device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        input:
            img: RGB img (batch x c x h x w)
        """
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = self.extractor_head(out)

        return out


class VisualController():

    def __init__(self, img_dim: int = 64, device='cuda') -> None:
        self.feature_extractor = VisualFeatureCNNExtractor(
            img_dim=img_dim, device=device)
        # TODO(jake): GP

    def control(self, img: torch.Tensor) -> Tuple:
        """[summary]

        Args:
            img (torch.Tensor): visual observation

        Returns:
            torch.Tensor: action
            torch.Tensor: uncertainty 
        """
        pass


if __name__ == "__main__":
    controller = VisualController()
