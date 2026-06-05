from torch import nn

from src.models.backbone import FeatureExtractor
from src.models.heads import CBAM, ConvHead, ClassificationHead, CornHead, corn_logits_to_label


class BCSModel(nn.Module):
    def __init__(self, backbone, attention, conv_head, classifier):
        super().__init__()

        self.net = nn.Sequential(
            backbone,
            attention,
            conv_head,
            classifier,
        )

    def forward(self, x):
        return self.net(x)


def build_model(config: dict) -> nn.Module:
    backbone = FeatureExtractor(pretrained=True)
    attention = CBAM(backbone.feature_channels)
    conv_head = ConvHead(backbone.feature_channels)

    num_classes = len(config["class_values"])
    if config["head"] == "corn":
        classifier = CornHead(conv_head.out_features, num_classes)
    else:
        classifier = ClassificationHead(conv_head.out_features, num_classes)

    return BCSModel(backbone, attention, conv_head, classifier)


def build_predictor(config: dict):
    if config["head"] == "corn":
        return corn_logits_to_label

    def argmax_predict(logits):
        return logits.argmax(dim=1)

    return argmax_predict
