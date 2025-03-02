# SpeechEmotionRecognition
#Overview
This project implements Speech Emotion Recognition (SER) using Artificial Neural Networks (ANN) and Mel-Frequency Cepstral Coefficients (MFCCs) as features.
The workflow involves:
Feature Extraction from audio files (MFCCs)
Model Training using a neural network
Model Evaluation to assess performance
Emotion Prediction from new audio samples
#Project Directory Structure
SER_Project/
├── dataset/                  # Audio dataset (sorted into emotion folders)
├── extracted_features/       # CSV file containing extracted MFCC features
├── models/                   # Trained models (.h5 files)
├── scripts/                  # Python scripts for processing and training
│   ├── data_preprocessing.py  # Extracts MFCC features
│   ├── train_model.py         # Trains ANN model
│   ├── evaluate_model.py      # Evaluates trained model
│   ├── predict_emotion.py     # Predicts emotion for new audio
│          
├── main.py                    # Automates the full workflow
├── requirements.txt            # Dependencies
└── README.md                   # Project Documentation
#Model Details
Uses ANN with 3 layers
Activation: ReLU, Softmax
Accuracy: ~85-90% (depends on dataset and training settings)
#Next Steps
Improve accuracy using CNNs or LSTMs
Deploy model as a Web API using FastAPI
Implement real-time emotion detection
