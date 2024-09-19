import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def _conv1d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )


class ResidualBlock1d(nn.Module):
    """The basic building block of a wide ResNet for 1D data"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
        inplace: bool = True,
    ) -> None:
        super(ResidualBlock1d, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = _conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = _conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        x = self.relu(x)

        return x

class ResNet1d_Noise(nn.Module):
    """Residual Networks for 1D data with Adaptive Gaussian Noise"""

    def __init__(
        self,
        classes: int,
        channel_dims: List[int],
        depths: List[int],
        in_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super(ResNet1d_Noise, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths
        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)
        self.conv = _conv1d(
            in_channels=in_channels,
            out_channels=channel_dims[0],
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm1d(channel_dims[0])

        self.stem = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(channel_dims[0], channel_dims[0], depths[0])
        self.layer2 = self._make_layer(channel_dims[0], channel_dims[1], depths[1], stride=2)
        self.layer3 = self._make_layer(channel_dims[1], channel_dims[2], depths[2], stride=2)
        self.layer4 = self._make_layer(channel_dims[2], channel_dims[3], depths[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channel_dims[3], classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
          
        if self.training:
            noise = torch.randn_like(x)
            x = x + noise
        
        x = self.layer4(x)
        x = self.avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels: int, out_channels: int, depth: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                _conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            ResidualBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                )
            )

        return nn.Sequential(*layers)

class ResNet1d_AdaptiveNoise(nn.Module):
    """Residual Networks for 1D data with Adaptive Gaussian Noise"""

    def __init__(
        self,
        classes: int,
        channel_dims: List[int],
        depths: List[int],
        in_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super(ResNet1d_AdaptiveNoise, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths
        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)
        self.conv = _conv1d(
            in_channels=in_channels,
            out_channels=channel_dims[0],
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm1d(channel_dims[0])

        self.stem = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(channel_dims[0], channel_dims[0], depths[0])
        self.layer2 = self._make_layer(channel_dims[0], channel_dims[1], depths[1], stride=2)
        self.layer3 = self._make_layer(channel_dims[1], channel_dims[2], depths[2], stride=2)
        self.layer4 = self._make_layer(channel_dims[2], channel_dims[3], depths[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channel_dims[3], classes)

    def forward(self, x: torch.Tensor, target: torch.Tensor, class_losses: dict) -> torch.Tensor:
        # Class별 손실에 따른 sigma 값을 설정
        class_sigmas = {class_idx: class_losses[class_idx].avg for class_idx in class_losses.keys()}
        
        # 각 데이터 포인트에 해당하는 클래스의 sigma 값을 가져옴
        sigma = torch.tensor([class_sigmas[int(t.item())] for t in target]).view(-1, 1, 1).to(x.device)
        
        x = self.stem(x)
        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)
                
        if self.training:
            noise = torch.randn_like(x) * sigma
            x = x + noise

        x = self.layer4(x)
        x = self.avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels: int, out_channels: int, depth: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                _conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            ResidualBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                )
            )

        return nn.Sequential(*layers)

class EuclideanResNet1d(nn.Module):
    """Residual Networks for 1D data"""

    def __init__(
        self,
        classes: int,
        channel_dims: List[int],
        depths: List[int],
        in_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super(EuclideanResNet1d, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths
        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)
        self.conv = _conv1d(
            in_channels=in_channels,
            out_channels=channel_dims[0],
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm1d(channel_dims[0])

        self.stem = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(channel_dims[0], channel_dims[0], depths[0])
        self.layer2 = self._make_layer(channel_dims[0], channel_dims[1], depths[1], stride=2)
        self.layer3 = self._make_layer(channel_dims[1], channel_dims[2], depths[2], stride=2)
        self.layer4 = self._make_layer(channel_dims[2], channel_dims[3], depths[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channel_dims[3], classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        # print("After stem:", x.shape)
        x = self.layer1(x)
        # print("After layer1:", x.shape)
        x = self.layer2(x)
        # print("After layer2:", x.shape)
        x = self.layer3(x)
        # print("After layer3:", x.shape)
        x = self.layer4(x)
        # print("After layer4:", x.shape)
        x = self.avg_pool(x).squeeze(-1)
        embedding = x.clone()
        # print("After avg_pool:", x.shape)
        x = self.fc(x)
        # print("After fc:", x.shape)
        return x, embedding

    def _make_layer(self, in_channels: int, out_channels: int, depth: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                _conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            ResidualBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                )
            )

        return nn.Sequential(*layers)

class EuclideanResNet1d_wo(nn.Module):
    """Residual Networks for 1D data"""

    def __init__(
        self,
        classes: int,
        channel_dims: List[int],
        depths: List[int],
        in_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super(EuclideanResNet1d_wo, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths
        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)
        self.conv = _conv1d(
            in_channels=in_channels,
            out_channels=channel_dims[0],
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm1d(channel_dims[0])

        self.stem = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(channel_dims[0], channel_dims[0], depths[0])
        self.layer2 = self._make_layer(channel_dims[0], channel_dims[1], depths[1], stride=2)
        self.layer3 = self._make_layer(channel_dims[1], channel_dims[2], depths[2], stride=2)
        self.layer4 = self._make_layer(channel_dims[2], channel_dims[3], depths[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channel_dims[3], classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        # print("After stem:", x.shape)
        x = self.layer1(x)
        # print("After layer1:", x.shape)
        x = self.layer2(x)
        # print("After layer2:", x.shape)
        x = self.layer3(x)
        # print("After layer3:", x.shape)
        x = self.layer4(x)
        # print("After layer4:", x.shape)
        embedding = x.clone()
        x = self.avg_pool(x).squeeze(-1)
        # print("After avg_pool:", x.shape)
        x = self.fc(x)
        # print("After fc:", x.shape)
        return x, embedding

    def _make_layer(self, in_channels: int, out_channels: int, depth: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                _conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            ResidualBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                )
            )

        return nn.Sequential(*layers)



class EuclideanResNet1d_Masking(nn.Module):
    """Residual Networks for 1D data"""

    def __init__(
        self,
        classes: int,
        channel_dims: List[int],
        depths: List[int],
        in_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super(EuclideanResNet1d_Masking, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths
        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)
        self.conv = _conv1d(
            in_channels=in_channels,
            out_channels=channel_dims[0],
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm1d(channel_dims[0])

        self.stem = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(channel_dims[0], channel_dims[0], depths[0])
        self.layer2 = self._make_layer(channel_dims[0], channel_dims[1], depths[1], stride=2)
        self.layer3 = self._make_layer(channel_dims[1], channel_dims[2], depths[2], stride=2)
        self.layer4 = self._make_layer(channel_dims[2], channel_dims[3], depths[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channel_dims[3], classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        # print("After stem:", x.shape)
        x = self.layer1(x)
        # print("After layer1:", x.shape)
        x = self.layer2(x)
        # print("After layer2:", x.shape)
        x = self.layer3(x)
        # print("After layer3:", x.shape)
        x = self.layer4(x)
        # print("After layer4:", x.shape)
        embedding = x.clone()
        x = self.avg_pool(x).squeeze(-1)
        # print("After avg_pool:", x.shape)
        x = self.fc(x)
        # print("After fc:", x.shape)
        return x, embedding

    def _make_layer(self, in_channels: int, out_channels: int, depth: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                _conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            ResidualBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                )
            )

        return nn.Sequential(*layers)



class EmbeddingToResNet1d(nn.Module):
    def __init__(self, embedding_dim: int, resnet: EuclideanResNet1d):
        super(EmbeddingToResNet1d, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Input shape:", x.shape)
        x = self.embedding(x)
        # print("After embedding:", x.shape)
        x = x.permute(0, 2, 1)
        # print("After permute:", x.shape)
        x = self.resnet(x)
        return x
    
class EmbeddingToResNet1dWithAdaptiveNoise(nn.Module):
    def __init__(self, embedding_dim: int, resnet: ResNet1d_AdaptiveNoise):
        super(EmbeddingToResNet1dWithAdaptiveNoise, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.resnet = resnet

    def forward(self, x: torch.Tensor, target: torch.Tensor, class_losses: dict) -> torch.Tensor:
        # Embedding step
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        # Adaptive Noise in ResNet
        x = self.resnet(x, target, class_losses)
        return x

class SupConResNet1d(nn.Module):
    """Supervised Contrastive Learning with ResNet1d backbone"""

    def __init__(self, resnet: EuclideanResNet1d, head: str = 'mlp', feat_dim: int = 128):
        super(SupConResNet1d, self).__init__()
        self.encoder = resnet
        dim_in = resnet.channel_dims[-1]

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f'Head not supported: {head}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class IncrementalEmbeddingToResNet1d(nn.Module):
    def __init__(self, embedding_dim: int, resnet: EuclideanResNet1d, initial_classes: int):
        super(IncrementalEmbeddingToResNet1d, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.resnet = resnet
        self.fc = nn.Linear(resnet.channel_dims[-1], initial_classes)
        self.classes = initial_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def add_classes(self, new_classes: int):
        new_fc = nn.Linear(self.resnet.channel_dims[-1], self.classes + new_classes)
        with torch.no_grad():
            new_fc.weight[:self.classes] = self.fc.weight
            new_fc.bias[:self.classes] = self.fc.bias
        self.fc = new_fc
        self.classes += new_classes


if __name__ == "__main__":
    # 임베딩 차원 설정
    embedding_dim = 256  # 임베딩 차원
    kernel_size = 27  # 원하는 커널 크기

    # EuclideanResNet1d 모델 초기화
    resnet = EuclideanResNet1d(
        classes=75,
        channel_dims=[64, 128, 256, 512],
        depths=[3, 4, 6, 3],
        in_channels=embedding_dim,  # 수정된 부분
        kernel_size=kernel_size,  # 커널 크기 설정
    )

    # Embedding을 적용한 모델 초기화
    model = EmbeddingToResNet1d(embedding_dim, resnet)

    # 임의의 (5, 512) 크기의 데이터 생성 (배치 크기 5, 길이 512, 값 범위 0~255)
    x = torch.randint(0, 256, (5, 512))

    # 모델에 데이터를 전달하여 순전파 수행
    output = model(x)

    # 출력 크기 확인
    print("Output shape:", output.shape)
