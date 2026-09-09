import mne
from load import load_subject

def epoch(raw, events):
    pass

def bandpass_filter(raw):
    pass

def normalize(epochs):
    pass

if __name__ == "__main__":
    run4, events4, run8, events8 = load_subject(
        "data/Subjects/S001R04.edf",
        "data/Subjects/S001R08.edf"
    )
    