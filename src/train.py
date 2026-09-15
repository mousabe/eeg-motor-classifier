import os

import torch
import torch.nn as nn

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from preprocess import prepare_dataset
from model import EEGCNN


def train():
    X, y, groups = prepare_dataset()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8
    )

    model = EEGCNN(
        input_channels=X.shape[1]
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 30

    os.makedirs(
        "models",
        exist_ok=True
    )

    best_accuracy = 0.0

    for epoch_number in range(epochs):
        model.train()

        total_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = correct / total
        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch_number + 1}/{epochs} "
            f"Loss: {average_loss:.4f} "
            f"Test Accuracy: {accuracy:.2%}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": X.shape[1],
                    "best_accuracy": best_accuracy,
                    "epoch": epoch_number + 1
                },
                "models/eeg_cnn.pt"
            )

            print(
                f"Best model saved "
                f"({best_accuracy:.2%})"
            )

    print(
        f"Training complete. "
        f"Best Test Accuracy: {best_accuracy:.2%}"
    )

    print(
        "Best model saved to models/eeg_cnn.pt"
    )


if __name__ == "__main__":
    train()