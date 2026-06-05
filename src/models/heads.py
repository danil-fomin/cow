import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        attention = self.mlp(self.average_pool(x)) + self.mlp(self.max_pool(x))
        return x * torch.sigmoid(attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        average_pooled = x.mean(dim=1, keepdim=True)
        max_pooled = x.max(dim=1, keepdim=True).values
        attention = self.conv(torch.cat([average_pooled, max_pooled], dim=1))
        return x * torch.sigmoid(attention)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()

        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)

        return x


def conv_norm_activation(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class ConvHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 512):
        super().__init__()

        self.conv_block = nn.Sequential(
            conv_norm_activation(in_channels, hidden_channels, kernel_size=1),
            conv_norm_activation(hidden_channels, hidden_channels, kernel_size=3),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features = hidden_channels

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)

        return torch.flatten(x, 1)


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CornHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()

        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes - 1),
        )

    def forward(self, x):
        return self.net(x)


def corn_logits_to_label(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = torch.cumprod(probabilities, dim=1)
    
    return (probabilities > 0.5).sum(dim=1)
