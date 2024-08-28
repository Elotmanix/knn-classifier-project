import numpy as np
from sklearn.preprocessing import StandardScaler


class KNNClassifier:
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = self.scaler.transform(X_test)
        y_pred = []
        for x in X_test:
            if self.metric == 'euclidean':
                distances = np.linalg.norm(self.X_train - x, axis=1)
            elif self.metric == 'manhattan':
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            else:
                raise ValueError("Unsupported metric. Use 'euclidean' or 'manhattan'.")
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            y_pred.append(np.bincount(nearest_labels).argmax())
        return np.array(y_pred)

