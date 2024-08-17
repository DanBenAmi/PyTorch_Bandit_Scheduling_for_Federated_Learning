import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleCNN(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(FlexibleCNN, self).__init__()

        input_channels = input_shape[0]

        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x32 or 28x28
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 32x32 or 28x28
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: 16x16 or 14x14

        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 16x16 or 14x14
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Output: 16x16 or 14x14
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: 8x8 or 7x7

        # Calculate the flattened size after convolution layers
        # Create a dummy input tensor to compute the output size after convolutions
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = self._forward_conv(dummy_input)
        self.flattened_size = dummy_output.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def _forward_conv(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Example usage for CIFAR-10 (32x32x3)
    model_cifar10 = FlexibleCNN(input_shape=(3, 32, 32), num_classes=10)

    # Example usage for Fashion MNIST (28x28x1)
    model_fashion_mnist = FlexibleCNN(input_shape=(1, 28, 28), num_classes=10)

    # Calculate the number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters in CIFAR-10 model: {count_parameters(model_cifar10)}")
    print(f"Number of parameters in Fashion MNIST model: {count_parameters(model_fashion_mnist)}")
