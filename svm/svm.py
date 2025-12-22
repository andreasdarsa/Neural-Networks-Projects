from time import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from pathlib import Path

# Load dataset
BASE_PATH = Path("data/UCI HAR Dataset")

# If we want to load the text data, we use a different value for the split parameter (split='test')
# The function loads the training data by default, hence why we set split='train'
def load_har(split='train'):
    X = np.loadtxt(BASE_PATH / split / f"X_{split}.txt")
    y = np.loadtxt(BASE_PATH / split / f"y_{split}.txt").astype(int)
    return X, y

X_train, y_train = load_har('train') # Load training data
X_test, y_test = load_har('test') # Load test data

print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
print(f'Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}')

# Standard scaler brings features to zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA to retain 90% variance while reducing dimensions to roughly 10% of the original features
pca = PCA(n_components=0.90, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f'Original number of features: {X_train.shape[1]}')
print(f'Number of features after PCA: {X_train_pca.shape[1]}')

# Evaluate the model on the test data while also calculating accuracy and training time
def evaluate_model(model, X_tr, y_tr, X_val, y_val):
    start = time()
    model.fit(X_tr, y_tr)
    end = time()
    train_time = (end - start) * 1000  # in milliseconds

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    return val_accuracy, train_time

results = []

# Linear SVM
linear_svm = SVC(kernel='linear', random_state=42)
val_acc, train_time = evaluate_model(linear_svm, X_train_pca, y_train, X_test_pca, y_test)
results.append(('Linear SVM', val_acc, train_time))

# An example of a correct and an example of an incorrect prediction
y_test_pred = linear_svm.predict(X_test_pca)
correct_indices = np.where(y_test_pred == y_test)[0]
incorrect_indices = np.where(y_test_pred != y_test)[0]
if len(correct_indices) > 0:
    print(f'Example of correct prediction - Index: {correct_indices[0]}, Predicted: {y_test_pred[correct_indices[0]]}, Actual: {y_test[correct_indices[0]]}')
if len(incorrect_indices) > 0:
    print(f'Example of incorrect prediction - Index: {incorrect_indices[0]}, Predicted: {y_test_pred[incorrect_indices[0]]}, Actual: {y_test[incorrect_indices[0]]}')

# RBF SVM
rbf_svm = SVC(kernel='rbf', random_state=42)
val_acc, train_time = evaluate_model(rbf_svm, X_train_pca, y_train, X_test_pca, y_test)
results.append(('RBF SVM', val_acc, train_time))

# Another example of correct and incorrect prediction for RBF SVM
y_test_pred_rbf = rbf_svm.predict(X_test_pca)
correct_indices_rbf = np.where(y_test_pred_rbf == y_test)[0]
incorrect_indices_rbf = np.where(y_test_pred_rbf != y_test)[0]
if len(correct_indices_rbf) > 0:
    print(f'RBF SVM - Example of correct prediction - Index: {correct_indices_rbf[0]}, Predicted: {y_test_pred_rbf[correct_indices_rbf[0]]}, Actual: {y_test[correct_indices_rbf[0]]}')
if len(incorrect_indices_rbf) > 0:
    print(f'RBF SVM - Example of incorrect prediction - Index: {incorrect_indices_rbf[0]}, Predicted: {y_test_pred_rbf[incorrect_indices_rbf[0]]}, Actual: {y_test[incorrect_indices_rbf[0]]}')

# 1-NN
knn = KNeighborsClassifier(n_neighbors=1)
val_acc, train_time = evaluate_model(knn, X_train_pca, y_train, X_test_pca, y_test)
results.append(('1-NN', val_acc, train_time))

# 3-NN
knn3 = KNeighborsClassifier(n_neighbors=3)
val_acc, train_time = evaluate_model(knn3, X_train_pca, y_train, X_test_pca, y_test)
results.append(('3-NN', val_acc, train_time))

# Nearest Centroid
nc = NearestCentroid()
val_acc, train_time = evaluate_model(nc, X_train_pca, y_train, X_test_pca, y_test)
results.append(('Nearest Centroid', val_acc, train_time))

# MLP network with hinge loss
mlp_hinge = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', max_iter=200, random_state=42)
val_acc, train_time = evaluate_model(mlp_hinge, X_train_pca, y_train, X_test_pca, y_test)
results.append(('MLP with Hinge Loss', val_acc, train_time))

# Display results
results_df = pd.DataFrame(results, columns=['Model', 'Validation Accuracy', 'Training Time (ms)'])
# Sort by Validation Accuracy descending
results_df = results_df.sort_values(by='Validation Accuracy', ascending=False).reset_index(drop=True)
print(results_df)
