#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load iris dataset
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int64)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a two-layer fully connected neural network (one hidden layer + output layer)
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # first layer
        self.relu = nn.ReLU()                          # activation
        self.fc2 = nn.Linear(hidden_size, num_classes) # second layer
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 4
hidden_size = 10
num_classes = 3

model = TwoLayerNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save learned weights to a text file in a simple format.
# Format:
#   W1 <rows> <cols>
#   [W1 rows...]
#   b1 <length>
#   [b1 vector]
#   W2 <rows> <cols>
#   [W2 rows...]
#   b2 <length>
#   [b2 vector]
with open("weights.txt", "w") as f:
    # Get weights and biases
    # Transpose fc1_weight to be 4x10 (instead of 10x4)
    fc1_weight = model.fc1.weight.detach().numpy().T
    fc1_bias   = model.fc1.bias.detach().numpy()
    # Transpose fc2_weight to be 10x3 (instead of 3x10)
    fc2_weight = model.fc2.weight.detach().numpy().T
    fc2_bias   = model.fc2.bias.detach().numpy()

    # Write fc1 weights and bias
    f.write(f"W1 {fc1_weight.shape[0]} {fc1_weight.shape[1]}\n")
    for row in fc1_weight:
        f.write(" ".join(map(str, row)) + "\n")
    f.write(f"b1 {len(fc1_bias)}\n")
    f.write(" ".join(map(str, fc1_bias)) + "\n")

    # Write fc2 weights and bias
    f.write(f"W2 {fc2_weight.shape[0]} {fc2_weight.shape[1]}\n")
    for row in fc2_weight:
        f.write(" ".join(map(str, row)) + "\n")
    f.write(f"b2 {len(fc2_bias)}\n")
    f.write(" ".join(map(str, fc2_bias)) + "\n")


# Save test data (features and labels) to CSV for inference in C.
# Each row: f1,f2,f3,f4,label
test_data = np.hstack((X_test, y_test.reshape(-1,1)))
np.savetxt("iris_test.csv", test_data, delimiter=",", header="f1,f2,f3,f4,label", comments="")

print("Training complete. Weights saved to 'weights.txt' and test data saved to 'iris_test.csv'.")
