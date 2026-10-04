import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.W = None
        self.b = None
        self.loss_history = []

    def softmax(self, Z):
        # Subtracting the row maximum improves numerical stability
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def train(self, X, y, num_classes, class_weights=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        n_samples, n_features = X.shape

        self.W = np.zeros((n_features, num_classes))
        self.b = np.zeros((1, num_classes))
        self.loss_history = []

        Y = np.eye(num_classes)[y]

        if class_weights is None:
            sample_weights = np.ones(n_samples)
        else:
            sample_weights = np.array([class_weights[label] for label in y])

        for i in range(self.iterations):
            Z = X @ self.W + self.b
            P = self.softmax(Z)

            # Weighted cross-entropy
            loss = -np.mean(
                sample_weights * np.sum(Y * np.log(P + 1e-15), axis=1)
            )
            self.loss_history.append(loss)

            # Weighted gradients
            error = (P - Y) * sample_weights[:, None]
            dW = (X.T @ error) / n_samples
            db = np.mean(error, axis=0, keepdims=True)

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self.softmax(X @ self.W + self.b)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)