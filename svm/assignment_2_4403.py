import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Initialize and train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Evaluate on test set
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')

print("Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Home Win', 'Draw', 'Away Win']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

