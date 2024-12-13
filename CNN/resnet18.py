import torchvision.models as models
import torch
import torch.nn as nn

# ResNet-18 model for CIFAR-10 (with modified final layer for 10 classes)
class ResNet18(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)

        # Modify the first conv layer to adapt to smaller input sizes like CIFAR-10
        self.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove maxpool layer since CIFAR-10 is already small
        self.model.maxpool = nn.Identity()

        # Modify the final fully connected layer
        self.model.fc = nn.Linear(512, num_classes)  # For 10 classes

    def forward(self, x):
        return self.model(x)