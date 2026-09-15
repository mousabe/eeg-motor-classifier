import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit

from preprocess import prepare_dataset
from model import EEGCNN


def evaluate():
    X, y, groups = prepare_dataset()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    _, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_test, y_test = X[test_idx], y[test_idx]

    checkpoint = torch.load(
        "models/eeg_cnn.pt",
        map_location="cpu"
    )

    model = EEGCNN(
        input_channels=checkpoint["input_channels"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    with torch.no_grad():
        outputs = model(X_tensor)

        predictions = outputs.argmax(
            dim=1
        ).numpy()

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Left",
                "Right"
            ]
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":
    evaluate()