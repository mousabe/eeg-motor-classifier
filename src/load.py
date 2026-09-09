import mne

def load_subject(run4_path, run8_path): 
    run4 = mne.io.read_raw_edf(str(run4_path), preload=True, verbose=False)
    run8 = mne.io.read_raw_edf(str(run8_path), preload=True, verbose=False)
    events4, _ = mne.events_from_annotations(run4)
    events8, _ = mne.events_from_annotations(run8)

    return run4, events4, run8, events8


if __name__ == "__main__":
    subject_1_run4 = "data/Subjects/S001R04.edf"
    subject_1_run8 = "data/Subjects/S001R08.edf"

    print(load_subject(subject_1_run4, subject_1_run8))