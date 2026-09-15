import time

import torch

from load import load_subject
from preprocess import prepare_run
from model import EEGCNN


def inference():
    run4, events4, _, _ = load_subject(
        "data/Subjects/S001R04.edf",
        "data/Subjects/S001R08.edf"
    )

    X, y = prepare_run(
        run4,
        events4
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

    for i, sample in enumerate(X):
        sample = torch.tensor(
            sample,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            output = model(sample)

            prediction = output.argmax(
                dim=1
            ).item()

        if prediction == 0:
            command = "MOVE LEFT"
        else:
            command = "MOVE RIGHT"

        actual = (
            "LEFT"
            if y[i] == 0
            else "RIGHT"
        )

        print(f"Prediction: {command} | " f"Actual: {actual}")

        time.sleep(1)


if __name__ == "__main__":
    inference()