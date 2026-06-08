import torch
from torch import nn
# from torchvision import models
from ultralytics import YOLO

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        # weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        # backbone = models.efficientnet_b0(weights=weights)

        # self.features = backbone.features
        # self.feature_channels = backbone.classifier[1].in_features

        yolo = YOLO("yolo26x-cls.pt")
        seq = yolo.model.model          
        self.features = seq[:-1]      

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            self.feature_channels = self.features(dummy).shape[1]

    def forward(self, x):
        return self.features(x)
