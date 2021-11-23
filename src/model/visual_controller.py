from __future__ import annotations
import torch
from torch import nn
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class VisualFeatureCNNExtractor(nn.Module):

    def __init__(self,
                 cnn_scaling: int = 1,
                 in_feats: int = 3,
                 img_dim: int = 64,
                 out_feat: int = 8,
                 device='cuda'):
        super().__init__()

        self.device = device
        self.img_dim = img_dim

        model = []
        """ Convolutional part """
        out_feats = cnn_scaling * in_feats
        num_deep_layers = 2
        for _ in range(num_deep_layers):
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
            out_feats *= cnn_scaling

        self.model = nn.Sequential(*model)

        in_feats_head = int(in_feats * (self.img_dim / num_deep_layers**2)**2)
        self.extractor_head = nn.Sequential(
            nn.Linear(in_features=in_feats_head, out_features=in_feats_head),
            nn.ReLU(),
            nn.Linear(in_features=in_feats_head, out_features=out_feat),
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
    Idea: use a pre-trained visual feature extractor?
        Ans: Not the best practice
    Idea: train a GP incrementally
        Ans: use model.set_train_data()
    """

    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood,
                 num_frames: int = 1,
                 img_dim: int = 64,
                 device='cuda') -> None:
        """
        Args:
            train_x (torch.Tensor): training images
            train_y (torch.Tensor): corresponding actions in demonstration
            likelihood ([type]): gpytorch.likelihoods.GaussianLikelihood()
            num_frames (int): number of input num_frames
            img_dim (int, optional): [description]. Defaults to 64.
            device (str, optional): [description]. Defaults to 'cuda'.
        """
        super(VisualGPController, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        self.img_dim = img_dim
        self.num_frames = num_frames
        self.feature_extractor = VisualFeatureCNNExtractor(
            in_feats=num_frames * 3, img_dim=img_dim, device=device)

        self.to(device)

    def forward(self,
                img: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Args:
            img (torch.Tensor): visual observation on device

        Returns:
            gpytorch distribution 
        """
        img = img.reshape(
            (img.shape[0], self.num_frames * 3, self.img_dim, self.img_dim))
        latent = self.feature_extractor(img)
        mean_a = self.mean_module(latent)
        covar_a = self.covar_module(latent)
        return gpytorch.distributions.MultivariateNormal(mean_a, covar_a)


class VisVarGPCtrler(gpytorch.models.ApproximateGP):
    """
    Idea: use a pre-trained visual feature extractor?
        Ans: Not the best practice
    Idea: train a GP incrementally
        Ans: use model.set_train_data()
    """

    def __init__(self,
                 inducing_points,
                 num_frames: int = 1,
                 img_dim: int = 64,
                 device='cuda'):
        feature_extractor = VisualFeatureCNNExtractor(
            in_feats=num_frames * 3, img_dim=img_dim, device=device, out_feat=4)

        inducing_points = feature_extractor(
            inducing_points.view(
                inducing_points.size(0), num_frames * 3, img_dim,
                img_dim).to(device))
        print("Inducing points shape:", inducing_points.shape)

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True)
        super(VisVarGPCtrler, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        self.feature_extractor = feature_extractor
        self.num_frames = num_frames
        self.img_dim = img_dim
        self.to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x):
        x = self.feature_extractor(
            x.view(x.size(0), self.num_frames * 3, self.img_dim, self.img_dim))
        return super().__call__(x)


class GPController(gpytorch.models.ExactGP):

    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood,
                 device='cuda') -> None:
        """
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
        """
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
