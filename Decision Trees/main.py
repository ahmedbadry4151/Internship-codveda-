# 1) Imports
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 2) Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train decision tree (tune max_depth to control complexity)
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,          # try None, 2, 3, 4...
    min_samples_leaf=2,   # simple pruning
    random_state=42
)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,          # color nodes by class/impurity
    rounded=True,
    proportion=True,      # show class proportions
    fontsize=10
)
plt.title("Decision Tree (Iris)")
plt.show()
