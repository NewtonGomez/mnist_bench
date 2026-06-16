import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.metrics import accuracy_score


class _Svm(nn.Module):
    """Internal module containing the trainable parameters of a binary SVM."""

    def __init__(self, num_samples):
        super().__init__()
        self.v = mx.zeros((num_samples,))
        self.b = mx.zeros((1,))

    def __call__(self, K):
        return K @ self.v + self.b


class SVC:
    """Binary SVM implemented in MLX.

    Maintained as the base classifier for multi-class strategies.
    For multi-class MNIST, use OneVsOneSVC instead.
    """

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        gamma=0.1,
        degree=3,
        lr=0.01,
        epochs=1000,
    ):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.lr = lr
        self.epochs = epochs

        self.model = None
        self.optimizer = None
        self.loss_and_grad_fn = None
        self.X_train = None
        self.y_train = None
        self.K_train = None
        self.history = {"loss": [], "accuracy": []}

    def _compute_kernel(self, X1, X2):
        """Vectorized kernel implementations."""
        if self.kernel_type == "linear":
            return X1 @ X2.T

        if self.kernel_type == "poly":
            return (self.gamma * (X1 @ X2.T) + 1.0) ** self.degree

        if self.kernel_type == "rbf":
            X1_sq = mx.sum(X1**2, axis=1, keepdims=True)
            X2_sq = mx.sum(X2**2, axis=1)
            squared_distances = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
            return mx.exp(-self.gamma * squared_distances)

        if self.kernel_type == "sigmoid":
            coef0 = 0.0
            return mx.tanh(self.gamma * (X1 @ X2.T) + coef0)

        if self.kernel_type == "laplacian":
            X1_exp = mx.expand_dims(X1, 1)
            X2_exp = mx.expand_dims(X2, 0)
            l1_distance = mx.sum(mx.abs(X1_exp - X2_exp), axis=-1)
            return mx.exp(-self.gamma * l1_distance)

        if self.kernel_type == "cosine":
            dot_product = X1 @ X2.T
            norm_X1 = mx.sqrt(mx.sum(X1**2, axis=1, keepdims=True))
            norm_X2 = mx.sqrt(mx.sum(X2**2, axis=1))
            return dot_product / ((norm_X1 @ mx.expand_dims(norm_X2, 0)) + 1e-8)

        raise ValueError(f"Kernel '{self.kernel_type}' is not supported.")

    @staticmethod
    def _loss_fn(model, K_mat, y_true, C_param):
        """Hinge loss + lightweight L2 regularization."""
        preds = model(K_mat)
        errors = mx.maximum(0.0, 1.0 - y_true * preds)
        hinge_loss = C_param * mx.mean(errors)

        Kv = K_mat @ model.v
        reg_loss = 0.5 * mx.mean(model.v * Kv)

        return hinge_loss + reg_loss

    def _prepare_training(self, X, y):
        """Initialize internal state for epoch-based training."""
        X = mx.array(X, dtype=mx.float32)
        y_np = np.where(np.array(y) <= 0, -1, 1).astype(np.float32)
        y = mx.array(y_np, dtype=mx.float32)

        self.X_train = X
        self.y_train = y
        self.K_train = self._compute_kernel(X, X)

        self.model = _Svm(X.shape[0])
        self.optimizer = optim.Adam(learning_rate=self.lr)
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

        mx.eval(self.model.parameters(), self.K_train)

    def train_epoch(self):
        """Execute a single training epoch and return the loss value."""
        if self.model is None:
            raise RuntimeError("Call _prepare_training() before train_epoch().")

        loss, grads = self.loss_and_grad_fn(
            self.model,
            self.K_train,
            self.y_train,
            self.C,
        )
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)

        return float(loss)

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=None,
        eval_every=1,
        verbose=True,
    ):
        """Train a binary SVM.

        Returns
        -------
        dict
            History dictionary with keys: ['loss', 'accuracy'].
        """
        if epochs is None:
            epochs = self.epochs

        self._prepare_training(X, y)
        history = {"loss": [], "accuracy": []}
        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            history["loss"].append(loss)

            should_eval = (
                X_val is not None
                and y_val is not None
                and (epoch == 1 or epoch % eval_every == 0 or epoch == epochs)
            )

            if should_eval:
                y_pred = self.predict(X_val)
                y_true = np.where(np.array(y_val) <= 0, -1, 1)
                accuracy = accuracy_score(y_true, y_pred)
                best_accuracy = max(best_accuracy, accuracy)
                history["accuracy"].append(float(accuracy))

                if verbose:
                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Loss: {loss:.4f} | "
                        f"Accuracy: {accuracy:.4f} | "
                        f"Best: {best_accuracy:.4f}"
                    )
            else:
                history["accuracy"].append(np.nan)

                if verbose:
                    print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.4f}")

        self.history = history
        return history

    def decision_function(self, X):
        """Return continuous decision scores before applying the sign function."""
        if self.model is None:
            raise RuntimeError("The model has not been trained.")

        X = mx.array(X, dtype=mx.float32)
        K_test = self._compute_kernel(X, self.X_train)
        preds = self.model(K_test)
        mx.eval(preds)

        return np.array(preds).flatten()

    def predict(self, X):
        """Binary classification: returns 1 or -1."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


class OneVsOneSVC:
    """Multi-class One-vs-One SVC for MNIST.

    Trains a binary classifier for each unique pair of classes.
    For MNIST, this yields 45 classifiers: C(10, 2).

    Its fit method returns a history dataset comparable to the neural network:
        dict_keys(['loss', 'accuracy'])

    - loss: Average loss of the 45 estimators per epoch.
    - accuracy: Ensembled multi-class accuracy over validation data.
    """

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        gamma=0.01,
        degree=3,
        lr=0.001,
        epochs=100,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.lr = lr
        self.epochs = epochs

        self.classes_ = None
        self.class_to_index_ = None
        self.models = []
        self.history = {"loss": [], "accuracy": []}

    def _make_binary_model(self):
        return SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            lr=self.lr,
            epochs=self.epochs,
        )

    def _prepare_pair_models(self, X, y):
        self.classes_ = np.unique(y)
        self.class_to_index_ = {
            label: idx for idx, label in enumerate(self.classes_)
        }
        self.models = []

        for pos_idx, pos_class in enumerate(self.classes_):
            for neg_class in self.classes_[pos_idx + 1 :]:
                mask = (y == pos_class) | (y == neg_class)
                X_pair = X[mask]
                y_pair = np.where(y[mask] == pos_class, 1, -1)

                model = self._make_binary_model()
                model._prepare_training(X_pair, y_pair)

                self.models.append((pos_class, neg_class, model))

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=None,
        eval_every=1,
        verbose=True,
    ):
        """Train the One-vs-One ensemble through synchronized epochs.

        Returns
        -------
        dict
            History dictionary with keys: ['loss', 'accuracy'].
        """
        if epochs is None:
            epochs = self.epochs

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
        if y_val is not None:
            y_val = np.asarray(y_val)

        self._prepare_pair_models(X, y)

        history = {"loss": [], "accuracy": []}
        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            epoch_losses = []

            for _, _, model in self.models:
                epoch_losses.append(model.train_epoch())

            avg_loss = float(np.mean(epoch_losses))
            history["loss"].append(avg_loss)

            should_eval = (
                X_val is not None
                and y_val is not None
                and (epoch == 1 or epoch % eval_every == 0 or epoch == epochs)
            )

            if should_eval:
                y_pred = self.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                best_accuracy = max(best_accuracy, accuracy)
                history["accuracy"].append(float(accuracy))

                if verbose:
                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"Accuracy: {accuracy:.4f} | "
                        f"Best: {best_accuracy:.4f}"
                    )
            else:
                history["accuracy"].append(np.nan)

                if verbose:
                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Avg Loss: {avg_loss:.4f}"
                    )

        self.history = history
        return history

    def predict(self, X):
        """Multi-class prediction using One-vs-One voting."""
        if not self.models:
            raise RuntimeError("The OneVsOneSVC model has not been trained.")

        X = np.asarray(X, dtype=np.float32)
        n_samples = len(X)
        votes = np.zeros((n_samples, len(self.classes_)), dtype=np.float32)
        row_idx = np.arange(n_samples)

        for pos_class, neg_class, model in self.models:
            preds = model.predict(X).flatten()

            pos_col = self.class_to_index_[pos_class]
            neg_col = self.class_to_index_[neg_class]

            votes[row_idx[preds == 1], pos_col] += 1
            votes[row_idx[preds == -1], neg_col] += 1

        return self.classes_[np.argmax(votes, axis=1)]