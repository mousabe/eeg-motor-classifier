import mne
import numpy as np

from load import load_subject


def bandpass_filter(raw):
    filtered = raw.copy()

    filtered.filter(
        l_freq=8,
        h_freq=30,
        verbose=False
    )

    return filtered


def epoch(raw, events):
    _, event_id = mne.events_from_annotations(
        raw,
        verbose=False
    )

    motor_event_id = {
        "T1": event_id["T1"],
        "T2": event_id["T2"]
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=motor_event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
        picks="eeg",
        verbose=False
    )

    return epochs, motor_event_id


def normalize(data):
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)

    std[std == 0] = 1

    return (data - mean) / std


def prepare_run(raw, events):
    raw = bandpass_filter(raw)

    epochs, event_id = epoch(
        raw,
        events
    )

    data = epochs.get_data(copy=True)

    data = normalize(data)

    labels = epochs.events[:, 2]

    labels = np.where(
        labels == event_id["T1"],
        0,
        1
    )

    return data.astype(np.float32), labels.astype(np.int64)


if __name__ == "__main__":
    run4, events4, run8, events8 = load_subject(
        "data/Subjects/S001R04.edf",
        "data/Subjects/S001R08.edf"
    )

    X4, y4 = prepare_run(
        run4,
        events4
    )

    X8, y8 = prepare_run(
        run8,
        events8
    )

    print("Run 4:", X4.shape, y4.shape)
    print("Run 8:", X8.shape, y8.shape)

    print("Run 4 labels:", y4)
    print("Run 8 labels:", y8)