import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from typing import Tuple, List
from KNNmodel import KNNClassifier

def create_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X, y = datasets.make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
        n_repeated=0, n_classes=2, random_state=42, class_sep=2
    )
    return X, y

def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=0.2, random_state=1234)

def evaluate_model(model: KNNClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, np.ndarray, str]:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    return accuracy, conf_matrix, class_report

def plot_decision_boundary(model: KNNClassifier, X: np.ndarray, y: np.ndarray, title: str) -> None:
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.savefig("KNN_decision_boundary.png", dpi=2000)
    

def main():
    # Create dataset
    X, y = create_dataset()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize and train model
    model = KNNClassifier(n_neighbors=5, metric='euclidean')
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Plot decision boundary
    plot_decision_boundary(model, X, y, "KNN Classification Decision Boundary")

if __name__ == "__main__":
    main()