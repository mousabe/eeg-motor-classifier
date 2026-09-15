import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from load import load_subject
from preprocess import prepare_run
from model import EEGCNN


def evaluate():
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
        X,
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
            y,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
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
            y,
            predictions
        )
    )


if __name__ == "__main__":
    evaluate()