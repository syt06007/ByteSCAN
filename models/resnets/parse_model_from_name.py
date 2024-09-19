import torch
import torch.nn as nn
import torch.nn.functional as F

from .euclidean1d import EuclideanResNet1d,EuclideanResNet1d_wo


class EmbeddingToResNet1d(nn.Module):
    def __init__(self, embedding_dim: int, resnet: EuclideanResNet1d):
        super(EmbeddingToResNet1d, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x, embedding = self.resnet(x)
        return x, embedding

class EmbeddingToResNet1dMasking(nn.Module):
    def __init__(self, embedding_dim: int, resnet: EuclideanResNet1d):
        super(EmbeddingToResNet1dMasking, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=257, embedding_dim=embedding_dim)
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x, embedding = self.resnet(x)
        return x, embedding


    
def parse_model_from_name(
    model_name: str,
    classes: int,
) -> nn.Module:
    # 기본값 설정
    default_channel_dims = [64, 128, 256, 512]
    default_depths = [2, 2, 2, 2]
    # default_depths = [3, 4, 6, 3]
    default_in_channels = 256  # 기본 임베딩 차원 설정
    kernel_size = 27

    # Check if the model_name is a simple type or a complex type
    if model_name in ["euclidean", "euclidean_without_avgpool", "euclidean_masking"]:
        if model_name == "euclidean":
            resnet = EuclideanResNet1d(
                classes=classes,
                channel_dims=default_channel_dims,
                depths=default_depths,
                in_channels=default_in_channels,
                kernel_size=kernel_size
            )
            return EmbeddingToResNet1d(default_in_channels, resnet)
        elif model_name =='euclidean_without_avgpool':
            resnet = EuclideanResNet1d_wo(
                classes=classes,
                channel_dims=default_channel_dims,
                depths=default_depths,
                in_channels=default_in_channels,
                kernel_size=kernel_size
            )
            return EmbeddingToResNet1d(default_in_channels, resnet)