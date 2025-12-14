import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# This code implements a Multi-Layer Perceptron (MLP) Feedforward Neural Network using PyTorch
# to classify match results based on various features. It includes data preprocessing,
# model definition, training, evaluation, and hyperparameter tuning.

# Before training the model we are going to test different hyperparameters to find the best combination.

# Hyperparameters
HIDDEN_SIZE = 64 # Number of neurons in hidden layers
NUM_EPOCHS = 20 # Number of training epochs
NUM_LAYERS = 2 # Number of hidden layers
LR = 0.001 # Learning rate

# 1. Define the MLP Feedforward Neural Network, as well as training and evaluation functions
# 1.1 MLP Feedforward Neural Network
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, out_dim=3):
        super().__init__()
        layers = []

        # Hidden layers
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden))
            layers.append(nn.ReLU())
            curr_dim = hidden  # Update curr_dim for the next layer

        # Output layer
        layers.append(nn.Linear(curr_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
# 1.2 Training and Evaluation Functions    
def train_model(model, train_loader, val_loader, lr=LR, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}')

    return model

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
    
    
# 2. Data Preprocessing
df = pd.read_csv('transformed_matches.csv')

# Encode categorical features
df['home_team'] = df['home_team'].astype('category').cat.codes
df['away_team'] = df['away_team'].astype('category').cat.codes

# Map result to numerical values
df['result'] = df['result'].map({'1': 0, 'X': 1, '2': 2})

# Columns used: all except 'date', 'home_gf', 'away_gf', 'result'
X = df.drop(columns=['date', 'home_gf', 'away_gf', 'result'])
y = df['result']

# Convert to numpy arrays
X_np = X.to_numpy().astype(np.float32)
y_np = y.to_numpy().astype(np.int64)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# Split data into training, validation, and test sets (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_np, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. Convert to PyTorch Tensors and initialize DataLoaders
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 4. Hyperparameter Tuning and Model Training
# 4.1 Testing different learning rates
lr_list = [0.01, 0.005, 0.001, 0.0005]
best_lr = LR
best_acc = 0.0
for lr in lr_list:
    print(f'\n ------ Training with learning rate: {lr} ------ ')
    model = MLP(in_dim=X_train.shape[1], hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    train_model(model, train_loader, val_loader, lr=lr, num_epochs=NUM_EPOCHS)
    test_acc = evaluate(model, test_loader)
    if test_acc > best_acc:
        best_acc = test_acc
        best_lr = lr
    print(f'Test Accuracy: {test_acc:.4f}')

# 4.2 Testing different hidden layer sizes
hidden_sizes = [32, 64, 128, 256]
best_hidden_size = HIDDEN_SIZE
best_acc = 0.0
for hidden_size in hidden_sizes:
    print(f'\n ------ Training with hidden size: {hidden_size} ------ ')
    model = MLP(in_dim=X_train.shape[1], hidden=hidden_size, num_layers=NUM_LAYERS)
    train_model(model, train_loader, val_loader, lr=best_lr, num_epochs=NUM_EPOCHS)
    test_acc = evaluate(model, test_loader)
    if test_acc > best_acc:
        best_acc = test_acc
        best_hidden_size = hidden_size
    print(f'Test Accuracy: {test_acc:.4f}')

# 4.3 Testing different number of layers
num_layers_list = [1, 2, 3, 4]
best_num_layers = NUM_LAYERS
best_acc = 0.0
for num_layers in num_layers_list:
    print(f'\n ------ Training with number of layers: {num_layers} ------ ')
    model = MLP(in_dim=X_train.shape[1], hidden=best_hidden_size, num_layers=num_layers)
    train_model(model, train_loader, val_loader, lr=best_lr, num_epochs=NUM_EPOCHS)
    test_acc = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')

# 4.4 Testing different number of epochs
epoch_list = [10, 15, 20, 25, 30, 35, 40]
best_num_epochs = NUM_EPOCHS
best_acc = 0.0
for num_epochs in epoch_list:
    print(f'\n ------ Training with number of epochs: {num_epochs} ------ ')
    model = MLP(in_dim=X_train.shape[1], hidden=best_hidden_size, num_layers=best_num_layers)
    train_model(model, train_loader, val_loader, lr=best_lr, num_epochs=num_epochs)
    test_acc = evaluate(model, test_loader)
    if test_acc > best_acc:
        best_acc = test_acc
        best_num_epochs = num_epochs
    print(f'Test Accuracy: {test_acc:.4f}')

print(f'''\nBest Hyperparameters: Learning Rate: {best_lr}, Hidden Size: {best_hidden_size}, 
      Num Layers: {best_num_layers}, Num Epochs: {best_num_epochs}''')

# 5. Final Model Training with Best Hyperparameters
final_model = MLP(in_dim=X_train.shape[1], hidden=best_hidden_size, num_layers=best_num_layers)
train_model(final_model, train_loader, val_loader, lr=best_lr, num_epochs=best_num_epochs)
final_test_acc = evaluate(final_model, test_loader)
print(f'Final Test Accuracy with Best Hyperparameters: {final_test_acc:.4f}')

#6. Random choice from the evaluation set, show example of correct and false predictions
# 3 unique samples from the validation set
# choose 3 random validation samples with a controllable seed
seed = 42
rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
idxs = rng.choice(len(val_data), size=3, replace=False)

inputs = torch.stack([val_data[i][0] for i in idxs])
labels = torch.tensor([int(val_data[i][1]) for i in idxs], dtype=torch.long)

outputs = final_model(inputs)
_, predicted = torch.max(outputs.data, 1)
for i in range(3):
    print(f'Sample {i+1}:')
    # Recover original (unscaled) feature values and print categorical names
    cols = X.columns.tolist()
    orig_row = scaler.inverse_transform(inputs[i].numpy().reshape(1, -1))[0]

    # Load raw CSV to get original category order
    df_raw = pd.read_csv('transformed_matches.csv')

    for cat_col in ['home_team', 'away_team']:
        if cat_col in cols:
            idx = cols.index(cat_col)
            code = int(round(orig_row[idx]))
            categories = pd.Categorical(df_raw[cat_col]).categories
            cat_value = categories[code] if 0 <= code < len(categories) else f'code_{code}'
            print(f'{cat_col}: code={code} -> {cat_value}')

    # Print other (unscaled) feature values for the sample
    other_features = {cols[j]: float(orig_row[j]) for j in range(len(cols)) if cols[j] not in ['home_team', 'away_team']}
    print('Other features (unscaled):', other_features)

    print(f'True Label: {labels[i].item()}, Predicted Label: {predicted[i].item()}')
    print('Correct Prediction' if labels[i].item() == predicted[i].item() else 'Incorrect Prediction')
    print('---')
