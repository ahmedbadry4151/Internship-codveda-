# Decision Tree Analysis on Iris Dataset

This document walks through training, evaluating, and visualizing a Decision Tree classifier on the Iris dataset.

## 1. Setup and Imports

First, we import the necessary libraries. We'll use `scikit-learn` for the model and data, and `matplotlib` for visualization.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 2. Load and Inspect Data

We load the classic Iris dataset.

```python
# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Convert to DataFrame for easier inspection (optional but recommended)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print("First 5 rows of the dataset:")
print(df.head())
```

## 3. Data Splitting

We split the data into training and testing sets. Using `stratify=y` ensures the class distribution is preserved in both sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")
```

## 4. Model Training

We initialize and train the Decision Tree. We use `max_depth` and `min_samples_leaf` to prevent overfitting (regularization).

```python
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,          # Limits tree depth
    min_samples_leaf=2,   # Requires at least 2 samples in a leaf
    random_state=42
)
clf.fit(X_train, y_train)
```

## 5. Evaluation

We evaluate the model using Accuracy, a Classification Report, and a Confusion Matrix.

```python
# Predictions
y_pred = clf.predict(X_test)

# Basic Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {acc:.3f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

## 6. Visualization

### Tree Structure
Visualizing the actual decision tree structure.

```python
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,          # Color nodes by class purity
    rounded=True,
    proportion=True,      # Show class proportions
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()
```

### Feature Importance
Understanding which features are most important for the decisions.

```python
import numpy as np

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices])
plt.tight_layout()
plt.show()
```
