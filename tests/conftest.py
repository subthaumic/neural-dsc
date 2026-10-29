import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from neural_dsc.mlp import MLP
import os

# Define the path for the cached model
# Model will be saved in the same directory as conftest.py (i.e., the 'tests' directory)
MODEL_DIR = os.path.dirname(__file__)
MODEL_FILENAME = "fixture_mlp_model.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


@pytest.fixture(scope="session")
def sample_data():
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)

    # Create simple test data
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


@pytest.fixture
def model(sample_data):
    mlp_model = MLP(layer_sizes=[4, 3, 3, 2])

    if os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH}")
        mlp_model.load_state_dict(torch.load(MODEL_PATH))
        mlp_model.eval()  # Set the model to evaluation mode
    else:
        print(f"Training model and saving to {MODEL_PATH}")
        # No need to create MODEL_DIR as it's the current directory (tests/)
        # os.makedirs(MODEL_DIR, exist_ok=True) # This line can be removed or commented out

        # Train the model
        mlp_model.fit(
            sample_data["train_loader"],
            nn.CrossEntropyLoss(),
            optim.SGD(mlp_model.parameters(), lr=0.01),
            num_epochs=10,
        )
        # Save the trained model
        torch.save(mlp_model.state_dict(), MODEL_PATH)
        mlp_model.eval()  # Also set to eval mode after training for consistency in tests

    return mlp_model
