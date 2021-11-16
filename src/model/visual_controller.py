from __future__ import annotations
import torch
from torch import nn
import gpytorch


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
            nn.Linear(in_features=in_feats_head, out_features=32),
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


class VisualGPController(gpytorch.models.ExactGP):
    """
    TODO: use a pre-trained visual feature extractor?
    TODO: ExactGP is not compatible with image inputs. 
        In exact_gp.py: 298, it requires test inputs to have the 
        same batch dim as the training inputs, but why???
    TODO: train a GP incrementally
    """

    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood,
                 img_dim: int = 64,
                 device='cuda') -> None:
        """[summary]

        Args:
            train_x (torch.Tensor): training images
            train_y (torch.Tensor): corresponding actions in demonstration
            likelihood ([type]): gpytorch.likelihoods.GaussianLikelihood()
            img_dim (int, optional): [description]. Defaults to 64.
            device (str, optional): [description]. Defaults to 'cuda'.
        """
        super(VisualGPController, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        self.feature_extractor = VisualFeatureCNNExtractor(
            img_dim=img_dim, device=device)

        self.to(device)

    def forward(self,
                img: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """[summary]

        Args:
            img (torch.Tensor): visual observation on device

        Returns:
            gpytorch distribution 
        """
        latent = self.feature_extractor(img)
        mean_a = self.mean_module(latent)
        covar_a = self.covar_module(latent)
        return gpytorch.distributions.MultivariateNormal(mean_a, covar_a)


class GPController(gpytorch.models.ExactGP):

    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood,
                 device='cuda') -> None:
        """[summary]

        Args:
            train_x (torch.Tensor): training states
            train_y (torch.Tensor): corresponding actions in demonstration
            likelihood ([type]): gpytorch.likelihoods.GaussianLikelihood()
            device (str, optional): [description]. Defaults to 'cuda'.
        """
        super(GPController, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        self.to(device)

    def forward(
            self,
            states: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """[summary]

        Args:
            states (torch.Tensor): input ground truth states

        Returns:
            gpytorch.distributions.MultivariateNormal: output distribution
        """
        mean_a = self.mean_module(states)
        covar_a = self.covar_module(states)
        return gpytorch.distributions.MultivariateNormal(mean_a, covar_a)


if __name__ == "__main__":
    controller = VisualGPController()
