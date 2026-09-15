import os

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from load import load_subject
from preprocess import prepare_run
from model import EEGCNN


def train():
    (run4, events4), (run8, events8) = load_subject(
        1,
        runs=[4, 8]
    )

    X4, y4 = prepare_run(
        run4,
        events4
    )

    X8, y8 = prepare_run(
        run8,
        events8
    )

    X = np.concatenate(
        [X4, X8],
        axis=0
    )

    y = np.concatenate(
        [y4, y8],
        axis=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

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