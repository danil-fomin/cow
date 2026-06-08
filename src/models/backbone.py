from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.feature_channels = backbone.classifier[1].in_features

    def forward(self, x):
        return self.features(x)
