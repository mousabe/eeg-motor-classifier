import shutil
from pathlib import Path

from mne.datasets import eegbci

RUNS = [4, 8, 12]
SUBJECTS = range(1, 11)

DATA_DIR = Path("data")
DEST_DIR = DATA_DIR / "Subjects"
DOWNLOAD_DIR = DATA_DIR / "raw_download"


def download_subjects(subjects, runs):
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        try:
            paths = eegbci.load_data(
                subject,
                runs,
                path=str(DOWNLOAD_DIR),
                update_path=False,
                verbose=False,
            )
        except Exception as e:
            print(f"Subject {subject:03d} failed: {e}")
            continue

        for path in paths:
            path = Path(path)
            shutil.move(str(path), str(DEST_DIR / path.name))

        print(f"Subject {subject:03d}: downloaded {len(paths)} files")

    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)


if __name__ == "__main__":
    download_subjects(SUBJECTS, RUNS)
