import torch
import torch.nn as nn
import torchvision.models as models


class SpatialModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ==================================================
        # Backbone
        # ==================================================
        base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # ==================================================
        # Feature Extractor
        # remove original classifier
        # ==================================================
        self.features = nn.Sequential(
            *list(base.children())[:-1]
        )

        # ==================================================
        # Classifier Head
        # ==================================================
        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 2)
        )

    def forward(self, x):

        # ==================================================
        # Feature Extraction
        # ==================================================
        x = self.features(x)

        # flatten
        x = torch.flatten(
            x,
            1
        )

        # ==================================================
        # Classification
        # ==================================================
        x = self.classifier(x)

        return x