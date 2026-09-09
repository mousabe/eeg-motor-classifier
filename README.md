# EEG Motor Classifier

A machine learning project that classifies imagined left-hand and right-hand movement from EEG brain signals.

The project loads EEG recordings, preprocesses the signals, trains a 1D CNN, and simulates real-time predictions that can be translated into robot arm commands.

## Project Structure

```text
eeg-motor-classifier/
│
├── data/
│   └── Subjects/
│       ├── S001R04.edf
│       └── S001R08.edf
│
├── src/
│   ├── load.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── results/
│   └── accuracy.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Files

### `src/load.py`
Loads the raw `.edf` EEG files and extracts event markers for left-hand and right-hand motor imagery.

### `src/preprocess.py`
Filters the EEG signals, splits them into epochs, and normalizes the data before training.

### `src/model.py`
Defines the PyTorch 1D CNN used to classify left-hand vs. right-hand motor imagery.

### `src/train.py`
Trains the neural network, evaluates its performance, and saves the trained model weights.

### `src/inference.py`
Loads the trained model and simulates real-time predictions by processing EEG windows one at a time.

### `notebooks/exploration.ipynb`
Used for plotting EEG signals, checking preprocessing results, and exploring the dataset.

### `data/`
Stores the raw EEG recordings used for training and testing.

### `results/`
Stores training graphs, accuracy plots, confusion matrices, and other model results.

## Pipeline

```text
EEG Signal
    ↓
Preprocessing
    ↓
1D CNN
    ↓
Left / Right Prediction
    ↓
Robot Arm Command
```