Horn Intent Recognition & Localization
**Project Name:** Research‑Project
**Author:** Sushanth Ganesh Maiya
**GitHub:** https://github.com/Sushanth‑Maiya/Research‑Project
**Contact:** sushanthmaiya98@gmail.com
————————————————————
Table of Contents
- Project Overview
- Features & Approach
- Repository Structure
- Environment & Installation
- Data Organization
- Feature Extraction
- Localization & GCC‑PHAT
- Training
- Inference / Prediction
- Evaluation
- Reproducibility & Notes
- Results (Example Placeholders)
- License & Citation
Project Overview
This repository implements a system for recognising horn sounds, classifying the intent (for example: warning, presence, frustration) and estimating the direction of the horn using stereo audio data (left + right channels). It combines classical machine‑learning feature‑extraction + Random Forest and LightGBM baselines, together with an exploratory convolutional neural network (CNN) on spectrograms for improved accuracy.
Features & Approach
- Stereo audio processing (two channels)
- Extraction of MFCCs, delta/delta‑delta features, spectral centroid, roll‑off, RMS, ZCR
- Localization using GCC‑PHAT to estimate direction based on inter‑channel time difference
- Random Forest and LightGBM models for intent classification
- CNN‑based spectrogram classification (exploratory)
- YAMNet‑based feature extraction for transfer learning
- Scripts for feature extraction, model training, prediction, and evaluation
- Designed with reproducibility and modularity in mind
Repository Structure

Research_Project/
└── Horn_Intent_recognition/
    ├── .vscode/                        # VSCode workspace settings
    ├── __pycache__/                    # Python cache files
    ├── audio/                          # Training Data
    ├── input/                          # Input stereo recordings
    │   ├── left_mic/                   # Left microphone channel data
    │   └── right_mic/                  # Right microphone channel data
    │
    ├── DetectHornP.py                  # Horn detection prediction script (variant 1)
    ├── DetectHornT.py                  # Horn detection prediction script (variant 2)
    ├── YAMNET_Labels.py                # Handles YAMNet label mapping
    ├── features.py                     # Feature extraction utilities (YAMNet-related)
    │
    ├── HornIntentPredictODAS.py        # Horn intent prediction using ODAS model
    ├── HornIntentPredictLGM.py         # Horn intent prediction using LightGBM model
    ├── HornIntentTrainODAS.py          # Training script for ODAS-based horn intent model
    ├── HornIntentTrainLGM.py           # Training script for LightGBM-based horn intent model
    │
    ├── horn_detector_model.pkl         # Trained horn detection model (pickle)
    ├── horn_detector_scalar.pkl        # Scaler for horn detection model
    ├── horn_intent_model.pkl           # Trained horn intent model
    │
    ├── horn_testin.py                  # Test script for horn detection logic
    ├── horndetectionYAMNET.py          # Horn detection using YAMNet (main detection model)
    ├── params.py                       # Parameter definitions for YAMNet and processing
    ├── predict_intent.py               # Main horn intent recognition + localization predictor
    ├── scalar.pkl                      # Main feature scaler (used in intent recognition)
    │
    ├── train_model.py                  # Main horn intent recognition + localization trainer
    ├── yamnet.h5                       # Pretrained YAMNet model weights
    ├── yamnet_class_map.csv            # YAMNet class mapping file (auto-generated)
    └── yamnet_model.py                 # YAMNet model definition script

Environment & Installation
This project uses a Conda environment for reproducibility. Create and activate using:
conda env create -f environment.yml
conda activate hornenvi
Data Organization
Place stereo .wav audio files and labels in the data/ directory or under Horn_Intent_recognition/input. Ensure consistent stereo channel order (left_mic and right_mic) and label format.
Feature Extraction
Feature extraction uses MFCCs, spectral centroid, rolloff, RMS, ZCR, and YAMNet embeddings. Scripts like features.py and horndetectionYAMNET.py handle feature computation and data preparation.
Localization & GCC‑PHAT
GCC‑PHAT is implemented to estimate the time delay between left and right channels, which is converted to angle estimation (in degrees) for sound localization.
Training
Training scripts include HornIntentTrainODAS.py, HornIntentTrainLGM.py, and train_model.py. These handle feature loading, dataset creation, and model training (Random Forest, LightGBM, ODAS, CNN).
Inference / Prediction
Prediction scripts include predict_intent.py, HornIntentPredictODAS.py, and HornIntentPredictLGM.py. They predict horn presence, intent, and direction using trained models.
Evaluation
Evaluation metrics include accuracy, precision, recall, F1-score, and localization error (MAE). Results and confusion matrices are saved in results/ or generated in script output.
Reproducibility & Notes
- Save the scaler used during training for reuse in inference.
- Maintain consistent stereo channel order across all recordings.
- Augment data with noise, time-shift, and amplitude scaling for robustness.
- Convert final models to TensorFlow Lite or ONNX for embedded inference.
- Keep environment.yml and requirements.txt updated for reproducibility.
Results (Example Placeholders)
- Random Forest (MFCC + GCC features): Accuracy = 0.92
- LightGBM (stereo features): Accuracy = 0.89
- CNN (spectrogram, exploratory): Validation Accuracy = 0.88
- Localization MAE: ~8° depending on mic baseline and noise
License & Citation
This project is licensed under the MIT License. If you reference or use this work, please cite:
Sushanth G. Maiya, Horn Intent Recognition & Localization (Research‑Project), GitHub repository, 2025.
