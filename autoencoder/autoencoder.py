import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# The autoencoder shall reconstruct the next digit in respect of the digit input.
# If the input is 0, the output should be 1; if the input is 1, the output should be 2; ...; if the input is 9, the output should be 0.

# 1. Data Loading and Preprocessing
digits = load_digits()
X = digits.data / 16.0  # Normalize pixel values to [0, 1]
y = digits.target

print("Data loaded.")

# Splitting the Data
X_input = []
y_target = []

for i in range(len(X) - 1):
    if y[i + 1] == y[i] + 1:
        X_input.append(X[i])
        y_target.append(X[i + 1])

X_input = np.array(X_input)
y_target = np.array(y_target)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_input, y_target, test_size=0.4, random_state=42
)
print("Data split into training and testing sets.")

# Displaying the Shapes of the Datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# 2. Autoencoder Model Definition
# Encoder, decoder, forward pass and latent_dim functions
def encoder(input_img, latent_dim):
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(latent_dim, activation='relu')(encoded)
    # All layers use ReLU activation except the output layer
    return encoded

def decoder(encoded_img, original_dim):
    decoded = Dense(64, activation='relu')(encoded_img)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(original_dim, activation='sigmoid')(decoded)
    # Output layer uses sigmoid activation to ensure outputs are in [0, 1]
    return decoded

def build_autoencoder(original_dim, latent_dim):
    input_img = Input(shape=(original_dim,))
    encoded = encoder(input_img, latent_dim)
    decoded = decoder(encoded, original_dim)
    autoencoder = Model(input_img, decoded)
    return autoencoder

# 3. Model Compilation and Training
# Loss, optimizer, epochs, loss per epoch, training time
original_dim = X_train.shape[1]
latent_dim = 32
autoencoder = build_autoencoder(original_dim, latent_dim)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary() # Display model architecture
print("Autoencoder model built and compiled.")

# Training the autoencoder
history = autoencoder.fit(X_train, y_train, 
                          epochs=50, 
                          batch_size=32, 
                          shuffle=True, 
                          validation_data=(X_test, y_test))
print("Autoencoder model trained.")

# 4. Model Evaluation
# Reconstructed next digit on test set, compute MSE, save reconstructed images
y_pred = autoencoder.predict(X_test)
mse = np.mean(np.square(y_test - y_pred))
print(f"Mean Squared Error on test set: {mse}")
# Save reconstructed images to a CSV file
reconstructed_df = pd.DataFrame(y_pred)
# Ensure the 'produced_data' directory exists before saving
os.makedirs('produced_data', exist_ok=True)
reconstructed_df.to_csv('produced_data/reconstructed_images.csv', index=False)
print("Reconstructed images saved to 'produced_data/reconstructed_images.csv'.")

# 5. PCA baseline
# fit PCA on training set, transform test set, compute MSE, same latent dimension
pca = PCA(n_components=latent_dim)
pca.fit(X_train)
X_test_pca_latent = pca.transform(X_test)
y_pca_reconstructed = pca.inverse_transform(X_test_pca_latent)
pca_mse = np.mean(np.square(y_test - y_pca_reconstructed))
print(f"PCA Mean Squared Error on test set: {pca_mse}")

# 6. Experiments
# Looping through different latent dimensions and recording MSEs, concentrated outputs
# latent dimensions to test
latent_dims = [8, 16, 32, 64, 128]
autoencoder_mse_results = {}
pca_mse_results = {}
for dim in latent_dims:
    # Autoencoder
    autoencoder = build_autoencoder(original_dim, dim)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, 
                    epochs=50, 
                    batch_size=32, 
                    shuffle=True, 
                    validation_data=(X_test, X_test), 
                    verbose=0)
    X_test_reconstructed = autoencoder.predict(X_test)
    mse = np.mean(np.square(X_test - X_test_reconstructed))
    autoencoder_mse_results[dim] = mse
    
    # PCA
    if dim > original_dim:
        dim = original_dim  # PCA cannot have more components than original features
    pca = PCA(n_components=dim)
    pca.fit(X_train)
    X_test_pca_reconstructed = pca.inverse_transform(pca.transform(X_test))
    pca_mse = np.mean(np.square(X_test - X_test_pca_reconstructed))
    pca_mse_results[dim] = pca_mse

# Display results
print("Autoencoder MSE results by latent dimension:")
for dim, mse in autoencoder_mse_results.items():
    print(f"Latent Dimension: {dim}, MSE: {mse}")
print("PCA MSE results by latent dimension:")
for dim, mse in pca_mse_results.items():
    print(f"Latent Dimension: {dim}, MSE: {mse}")

print("Experiments completed.")

# 7. Visualization
# Input vs Actual vs Predicted for a few samples
# Loss curves for training and validation
# Good/bad reconstructions
num_samples = 5
fig, axes = plt.subplots(num_samples, 3, figsize=(9, 8))
for i in range(num_samples):
    # Original
    axes[i, 0].imshow(X_test[i].reshape(8, 8), cmap='gray')
    axes[i, 0].set_title('Input')
    axes[i, 0].axis('off')
    
    # Autoencoder Reconstruction
    axes[i, 1].imshow(y_test[i].reshape(8, 8), cmap='gray')
    axes[i, 1].set_title('Actual')
    axes[i, 1].axis('off')
    
    # PCA Reconstruction
    axes[i, 2].imshow(y_pred[i].reshape(8, 8), cmap='gray')
    axes[i, 2].set_title('Predicted')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('img/reconstructions.png')
plt.show()

# Plotting loss curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('img/loss_curves.png')
plt.show()

print("Visualization completed.")

