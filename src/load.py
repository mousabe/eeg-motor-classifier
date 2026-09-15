from pathlib import Path

import mne

DATA_DIR = Path("data/Subjects")
RUNS = [4, 8, 12]


def run_path(subject, run):
    return DATA_DIR / f"S{subject:03d}R{run:02d}.edf"


def load_run(subject, run):
    raw = mne.io.read_raw_edf(str(run_path(subject, run)), preload=True, verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)

    return raw, events


def load_subject(subject, runs=RUNS):
    return [load_run(subject, run) for run in runs]


if __name__ == "__main__":
    runs = load_subject(1)

    for run_number, (raw, events) in zip(RUNS, runs):
        print(f"Run {run_number}:", raw)
        print(f"Run {run_number} events:", events)
