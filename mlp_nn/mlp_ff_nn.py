import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# 1. Define the MLP Feedforward Neural Network
class MLPFeedforwardNN(nn.Module):
    def __init__(self, in_dim=10, hidden=64, out_dim=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.layers(x)
    
# 2. Data Preprocessing
df = pd.read_csv('transformed_matches.csv')

# Encode categorical features
df['home_team'] = df['home_team'].astype('category').cat.codes
df['away_team'] = df['away_team'].astype('category').cat.codes

df['result'] = df['result'].map({'1': 0, 'X': 1, '2': 2})

X = df.drop(columns=['date', 'home_gf', 'away_gf', 'result'])
y = df['result']

X_np = X.to_numpy().astype(np.float32)
y_np = y.to_numpy().astype(np.int64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_np, test_size=0.3, random_state=42)

# 3. Convert to PyTorch Tensors and initialize DataLoaders
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 4. Train loop
HIDDEN_SIZE = 64
NUM_EPOCHS = 25
LR = 0.001


model = MLPFeedforwardNN(in_dim=X_train.shape[1], hidden=HIDDEN_SIZE, out_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}')

# 5. Evaluate the model
def accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

train_acc = accuracy(train_loader)
test_acc = accuracy(test_loader)

print(f'Using {NUM_EPOCHS} epochs, hidden size of {HIDDEN_SIZE}, and learning rate of {LR}')
print(f'Train Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
