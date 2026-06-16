import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    """
    Multilayer Perceptron con número variable de capas ocultas.
    """

    def __init__(
        self,
        in_features=784,
        hidden_dims=[384, 128, 64],
        num_classes=10,
        dropout_prob=0.5
    ):
        super().__init__()

        self.layers = []
        self.drop = nn.Dropout(p=dropout_prob)

        prev_dim = in_features

        # Crear automáticamente las capas ocultas
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Capa de salida
        self.out = nn.Linear(prev_dim, num_classes)

    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)
            x = mx.maximum(0.0, x)  # ReLU
            x = self.drop(x)

        return self.out(x)
    
def mlp_cross_entropy(model, X, y):
    """
    Wrapper function to compute the mean cross-entropy loss.
    Matches the signature expected by nn.value_and_grad: (model, X, y).
    """
    # 1. Perform the forward pass to get the raw logits
    logits = model(X)
    
    # 2. Compute the cross entropy between logits and true labels
    # MLX returns a loss value per sample in the batch, so we take the mean
    return mx.mean(nn.losses.cross_entropy(logits, y))