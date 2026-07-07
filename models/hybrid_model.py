import torch
import torch.nn as nn
import torchvision.models as models


class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ==================================================
        # Spatial Backbone
        # ==================================================
        spatial_base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.spatial = nn.Sequential(
            *list(spatial_base.children())[:-1]
        )

        # ==================================================
        # Frequency Backbone
        # ==================================================
        frequency_base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.frequency = nn.Sequential(
            *list(frequency_base.children())[:-1]
        )

        # ==================================================
        # Fusion Classifier
        # ==================================================
        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 2)
        )

    def forward(self, rgb, fft):

        # ==================================================
        # Spatial Features
        # ==================================================
        spatial_features = self.spatial(rgb)

        spatial_features = torch.flatten(
            spatial_features,
            1
        )

        # ==================================================
        # Frequency Features
        # ==================================================
        frequency_features = self.frequency(fft)

        frequency_features = torch.flatten(
            frequency_features,
            1
        )

        # ==================================================
        # Feature Fusion
        # ==================================================
        fused = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )

        # ==================================================
        # Classification
        # ==================================================
        output = self.classifier(fused)

        return output