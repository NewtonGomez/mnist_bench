import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.metrics import accuracy_score


class MLP(nn.Module):
    """Multilayer Perceptron for MNIST classification.

    Practical API:
        model.fit(...)
        model.predict(...)

    The fit method returns a comparable history:
        dict_keys(['loss', 'accuracy'])
    """

    def __init__(
        self,
        in_features=784,
        hidden_dims=None,
        num_classes=10,
        dropout_prob=0.2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [384, 128, 64]

        self.layers = []
        self.drop = nn.Dropout(p=dropout_prob)

        prev_dim = in_features

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.out = nn.Linear(prev_dim, num_classes)
        self.history = {"loss": [], "accuracy": []}

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = mx.maximum(0.0, x)  # ReLU
            x = self.drop(x)

        return self.out(x)

    def loss_fn(self, X, y):
        """Average cross-entropy for multi-class classification."""
        logits = self(X)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    def batch_iter(self, X, y, batch_size=128, shuffle=True):
        """Internal mini-batch generator."""
        n = len(X)
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            X_batch = mx.array(X[batch_idx], dtype=mx.float32)
            y_batch = mx.array(y[batch_idx], dtype=mx.int32)

            yield X_batch, y_batch

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        learning_rate=0.001,
        batch_size=128,
        eval_every=1,
        verbose=True,
    ):
        """Train the model using Adam.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training features.
        y_train : numpy.ndarray
            Training labels.
        X_val : numpy.ndarray, optional
            Validation features.
        y_val : numpy.ndarray, optional
            Validation labels.
        epochs : int, default=100
            Number of training epochs.
        learning_rate : float, default=0.001
            Learning rate for the optimizer.
        batch_size : int, default=128
            Size of mini-batches.
        eval_every : int, default=1
            Epoch interval to run validation.
        verbose : bool, default=True
            Whether to print progress metrics.

        Returns
        -------
        dict
            History dictionary with the format:
            {
                'loss': [loss_epoch_1, ...],
                'accuracy': [acc_epoch_1, ...]
            }

        Note: If eval_every > 1, np.nan is saved in un-evaluated epochs
        to keep loss and accuracy at the same length.
        """
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)

        if X_val is not None:
            X_val = X_val.astype(np.float32)
        if y_val is not None:
            y_val = y_val.astype(np.int32)

        mx.eval(self.parameters())

        optimizer = optim.Adam(learning_rate=learning_rate)
        loss_and_grad_fn = nn.value_and_grad(
            self,
            lambda model, X, y: model.loss_fn(X, y),
        )

        history = {"loss": [], "accuracy": []}
        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_losses = []

            for X_batch, y_batch in self.batch_iter(
                X_train,
                y_train,
                batch_size=batch_size,
                shuffle=True,
            ):
                loss, grads = loss_and_grad_fn(self, X_batch, y_batch)
                optimizer.update(self, grads)
                mx.eval(self.parameters(), optimizer.state, loss)
                epoch_losses.append(float(loss))

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
                        f"Loss: {avg_loss:.4f} | "
                        f"Accuracy: {accuracy:.4f} | "
                        f"Best: {best_accuracy:.4f}"
                    )
            else:
                history["accuracy"].append(np.nan)

                if verbose:
                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Loss: {avg_loss:.4f}"
                    )

        self.history = history
        return history

    def predict(self, X, batch_size=512):
        """Generate multi-class predictions in batches."""
        self.eval()

        X = X.astype(np.float32)
        preds = []

        for start in range(0, len(X), batch_size):
            end = start + batch_size
            X_batch = mx.array(X[start:end], dtype=mx.float32)

            logits = self(X_batch)
            y_pred = mx.argmax(logits, axis=1)
            mx.eval(y_pred)

            preds.append(np.array(y_pred))

        return np.concatenate(preds)