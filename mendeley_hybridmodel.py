import torch
import torch.nn as nn
import torchvision.models as models

class HybridModel(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridModel, self).__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Identity()

        # quantum layer
        self.quantum = nn.Module()
        self.quantum.linear = nn.Linear(512, 128)

        # classifier 
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),   
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  
        )

    def forward(self, x):
        x = self.base_model(x)

        # quantum layer
        x = self.quantum.linear(x)
        x = torch.relu(x)

        # classifier
        x = self.classifier(x)

        return x