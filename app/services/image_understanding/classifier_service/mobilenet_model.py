from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from .model_paths import classifier_weight_path
except ImportError:
    from model_paths import classifier_weight_path


MODULE_DIR = Path(__file__).resolve().parent
BACKBONE_WEIGHTS = classifier_weight_path("mobilenet_v3_large-8738ca79.pth")


def _load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        state_dict = _load_torch_file(BACKBONE_WEIGHTS)
        self.backbone.load_state_dict(state_dict)
        self.backbone.classifier[3] = nn.Linear(
            self.backbone.classifier[3].in_features,
            num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(weights_path: Path, num_classes: int):
    model = MobileNetV3Classifier(num_classes=num_classes)
    checkpoint = _load_torch_file(Path(weights_path).resolve())
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
