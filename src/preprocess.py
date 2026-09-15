import mne
import numpy as np

from load import RUNS, load_subject

SUBJECTS = range(1, 11)


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


def prepare_subject(subject, runs=RUNS):
    X_runs = []
    y_runs = []

    for raw, events in load_subject(subject, runs):
        data, labels = prepare_run(raw, events)

        X_runs.append(data)
        y_runs.append(labels)

    return np.concatenate(X_runs), np.concatenate(y_runs)


def prepare_dataset(subjects=SUBJECTS, runs=RUNS):
    X_subjects = []
    y_subjects = []
    groups_subjects = []

    for subject in subjects:
        X, y = prepare_subject(subject, runs)

        X_subjects.append(X)
        y_subjects.append(y)
        groups_subjects.append(np.full(len(y), subject, dtype=np.int64))

    X = np.concatenate(X_subjects)
    y = np.concatenate(y_subjects)
    groups = np.concatenate(groups_subjects)

    return X, y, groups


if __name__ == "__main__":
    X, y, groups = prepare_dataset()

    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)
    print("Subjects:", np.unique(groups))
    print("Class balance:", np.bincount(y))
