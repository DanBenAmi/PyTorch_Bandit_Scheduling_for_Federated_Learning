import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Generate some example data
x_train = torch.randn(70000, 10)  # 1000 samples, 10 features
true_weights = torch.arange(10).float() / 10
y_train = torch.matmul(x_train, true_weights) + 2 + 0.05 * torch.randn(70000)  # Linear relation with noise

# Normalize the data
x_mean = x_train.mean(dim=0, keepdim=True)
x_std = x_train.std(dim=0, keepdim=True)
x_train_normalized = (x_train - x_mean) / x_std

y_mean = y_train.mean()
y_std = y_train.std()
y_train_normalized = (y_train - y_mean) / y_std

# Reshape y_train to match the output dimension
y_train_normalized = y_train_normalized.view(-1, 1)

# Define the model, loss function, and optimizer
input_dim = 10
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    outputs = model(x_train_normalized)
    loss = criterion(outputs, y_train_normalized)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()  # Compute the gradients
    optimizer.step()  # Update the weights

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extract model parameters
learned_weights = model.linear.weight.data
learned_bias = model.linear.bias.data

# Denormalize weights and bias
denormalized_weights = learned_weights / x_std * y_std
denormalized_bias = (learned_bias * y_std + y_mean) - (learned_weights @ x_mean.t() / x_std * y_std)

# Print denormalized weights and bias
print("Learned weights (denormalized):", denormalized_weights)
print("Learned bias (denormalized):", denormalized_bias)

# Making predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    predicted = model(x_train_normalized).detach().numpy()

# Print some of the predictions
print("Predictions (first 5):", predicted[:5])
