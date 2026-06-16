import os
import sys
import mlx.core as mx
import numpy as np
from sklearn.metrics import accuracy_score

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.svm import SVC
from src.tools import load_flatten_mnist

from argos.callbacks import ArgosGuard

if __name__ == "__main__":
    EPOCHS = 800
    LEARNING_RATE = 0.001  # Increased slightly because the problem is easier
    MAX_SAMPLES = 15000

    print("Loading and normalizing the MNIST dataset...")
    X_train, y_train, x_test, y_test = load_flatten_mnist(
        base_path="data/raw/"
    )

    # Random Subsampling
    
    print(f"Reducing training data to {MAX_SAMPLES} random samples...")
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    indices = np.random.permutation(len(X_train_np))
    X_train_sub = X_train_np[indices[:MAX_SAMPLES]]
    y_train_sub = y_train_np[indices[:MAX_SAMPLES]]

    
    # Train 45 SVM Models (One-vs-One)
    
    ovo_models = []
    argos = ArgosGuard(prefix="SVC_test",check_memory=True)

    print("\nStarting One-vs-One training (45 models in total)...")

    contador_modelos = 0

    # Nested loop to create all unique pairwise combinations without repetition
    for i in range(10):
        for j in range(i + 1, 10):
            print(f"Training model: {i} vs {j}...")

            # Filter the dataset to keep ONLY the images of digits 'i' and 'j'
            mask = (y_train_sub == i) | (y_train_sub == j)
            X_pair = X_train_sub[mask]
            y_pair_original = y_train_sub[mask]

            # Create binary labels (i = 1, j = -1)
            y_pair_binary = np.where(y_pair_original == i, 1, -1)

            # Instantiate the model (using an RBF kernel)
            svm_mlx = SVC(
                kernel="rbf",
                gamma=0.01,
                C=1.0,
                lr=LEARNING_RATE,
                epochs=EPOCHS,
            )

            # Train the SVM with this small subset (very fast on Apple Silicon)
            svm_mlx.fit(X_pair, y_pair_binary)

            # Store a tuple with the model metadata: (positive, negative, model)
            ovo_models.append((i, j, svm_mlx))

            contador_modelos += 1
            # Argos avisa cada 10 modelos para que sepas que sigue vivo
            if contador_modelos % 10 == 0:
                argos.send_heartbeat(f"Completados {contador_modelos}/45 modelos (One-vs-One).")


    
    # Prediction and Voting (One-vs-One)
    
    print("\nEvaluating the One-vs-One ensemble on the full test set...")

    # Matrix to store votes. Rows: 10,000 images, Columns: 10 possible digits
    votes = np.zeros((len(x_test), 10))

    for i, j, model in ovo_models:
        # The model casts its vote (1 in favor of 'i', -1 in favor of 'j')
        preds = np.array(model.predict(x_test)).flatten()

        # Count the votes using highly optimized vectorized logic
        # If it predicted 1, add a point to the counter for digit 'i'
        votes[np.arange(len(x_test))[preds == 1], i] += 1
        # If it predicted -1, add a point to the counter for digit 'j'
        votes[np.arange(len(x_test))[preds == -1], j] += 1
        

    # The digit that accumulated the most votes for each image is the prediction
    final_predictions = np.argmax(votes, axis=1)

    # Evaluate the final result
    accuracy = accuracy_score(y_test, final_predictions)
    print(
        f"\nProcess finished! Test Accuracy (One-vs-One): {accuracy * 100:.2f}%"
    )
    argos.on_process_end(f"Test Accuracy: {accuracy * 100:.2f}%")