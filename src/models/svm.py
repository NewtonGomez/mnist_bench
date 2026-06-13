import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class _Svm(nn.Module):
    """Internal module containing the trainable SVM parameters."""

    def __init__(self, num_samples):
        super().__init__()
        # 'v' represents the coefficients for each support vector
        self.v = mx.zeros((num_samples,))
        # 'b' represents the bias
        self.b = mx.zeros((1,))

    def __call__(self, K):
        # Prediction equation in the kernel space: K @ v + b
        return K @ self.v + self.b


class SVC:
    """Support Vector Machine classifier implemented.

    Designed with a scikit-learn style API.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.1, degree=3, lr=0.01, epochs=1000):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.lr = lr
        self.epochs = epochs

        self.model = None
        self.X_train = None

    def _compute_kernel(self, X1, X2):
        """Vectorized implementation of kernel tricks."""
        if self.kernel_type == "linear":
            return X1 @ X2.T

        elif self.kernel_type == "poly":
            # K(x, y) = (gamma * <x, y> + 1)^degree
            return (self.gamma * (X1 @ X2.T) + 1.0) ** self.degree

        elif self.kernel_type == "rbf":
            # K(x, y) = exp(-gamma * ||x - y||^2)
            # Binomial expansion for matrix optimization
            X1_sq = mx.sum(X1**2, axis=1, keepdims=True)
            X2_sq = mx.sum(X2**2, axis=1)
            squared_distances = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
            return mx.exp(-self.gamma * squared_distances)

        elif self.kernel_type == "sigmoid":
            # K(x, y) = tanh(gamma * <x, y> + coef0)
            # Note: self.coef0 would need to be added to __init__ (usually 0.0)
            coef0 = 0.0
            return mx.tanh(self.gamma * (X1 @ X2.T) + coef0)

        elif self.kernel_type == "laplacian":
            # K(x, y) = exp(-gamma * ||x - y||_1)
            # Compute Manhattan Distance (L1) using broadcasting

            # Expand dimensions to compare each point against all others
            # X1 becomes (N, 1, D) and X2 becomes (1, M, D)
            X1_exp = mx.expand_dims(X1, 1)
            X2_exp = mx.expand_dims(X2, 0)

            # Sum of absolute differences
            l1_distance = mx.sum(mx.abs(X1_exp - X2_exp), axis=-1)
            return mx.exp(-self.gamma * l1_distance)

        elif self.kernel_type == "cosine":
            # K(x, y) = <x, y> / (||x|| * ||y||)
            # Compute dot product
            dot_product = X1 @ X2.T

            # Compute vector norms (magnitudes)
            norm_X1 = mx.sqrt(mx.sum(X1**2, axis=1, keepdims=True))
            norm_X2 = mx.sqrt(mx.sum(X2**2, axis=1))

            # Multiply norms and divide
            # Add 1e-8 to prevent division by zero
            return dot_product / (
                (norm_X1 @ mx.expand_dims(norm_X2, 0)) + 1e-8
            )
        else:
            raise ValueError(f"Kernel '{self.kernel_type}' is not supported.")

    def decision_function(self, X):
        """Return the raw continuous prediction values before applying the sign.

        Crucial for breaking ties in One-vs-Rest strategies.
        """
        if self.model is None:
            raise Exception("The model has not been trained.")

        X = mx.array(X)
        K_test = self._compute_kernel(X, self.X_train)
        preds = self.model(K_test)

        # Convert the MLX tensor to a NumPy array and flatten it
        return np.array(preds).flatten()

    def fit(self, X, y):
        """Fit the SVM model according to the given training data."""
        # Data preparation (convert labels to -1 and 1)
        X = mx.array(X)
        y_np = np.where(np.array(y) <= 0, -1, 1)
        y = mx.array(y_np, dtype=mx.float32)

        self.X_train = X
        num_samples = X.shape[0]

        # Precompute the training kernel matrix (K)
        K = self._compute_kernel(X, X)

        # Initialize the MLX model and optimizer
        self.model = _Svm(num_samples)
        optimizer = optim.Adam(learning_rate=self.lr)

        # Define the loss function (Regularization + Hinge Loss)
        def loss_fn(model, K_mat, y_true, C_param):
            """Compute the SVM loss combining Hinge Loss and light L2 regularization."""
            # Predictions
            preds = model(K_mat)

            # Hinge Loss (the pure classification error)
            errors = mx.maximum(0.0, 1.0 - y_true * preds)
            hinge_loss = C_param * mx.mean(
                errors
            )  # Use mean to remain independent of dataset size

            # Light L2 Regularization (prevents weights from exploding)
            Kv = K_mat @ model.v
            reg_loss = 0.5 * mx.mean(model.v * Kv)

            # Return the combined loss sum
            return hinge_loss + reg_loss

        # Create the function that returns loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        # Training loop (Gradient Descent)
        for epoch in range(self.epochs):
            loss, grads = loss_and_grad_fn(self.model, K, y, self.C)
            optimizer.update(self.model, grads)

            # Lazy evaluation characteristic of MLX
            mx.eval(self.model.parameters(), optimizer.state)

            if epoch % 200 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

        return self

    def predict(self, X):
        """Perform classification on samples in X."""
        if self.model is None:
            raise Exception(
                "The model has not been trained yet. Call .fit() first."
            )

        X = mx.array(X)
        # Evaluate the kernel of the new data against the training data
        K_test = self._compute_kernel(X, self.X_train)

        # Predict using the model
        preds = self.model(K_test)

        # Convert continuous predictions to classes (-1 or 1)
        # and return as a NumPy array
        return np.where(np.array(preds) >= 0, 1, -1)