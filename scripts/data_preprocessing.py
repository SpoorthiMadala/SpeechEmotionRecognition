import os
import librosa
import numpy as np
import pandas as pd

# Define dataset path correctly
DATASET_PATH = r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\dataset"

# List of emotions
EMOTIONS = ["happy", "sad", "angry", "calm", "surprised", "disgust", "fearful", "neutral"]

# Function to load audio
def load_audio(file_path, sr=22050):
    signal, _ = librosa.load(file_path, sr=sr)
    return signal

# Function to extract MFCC features
def extract_mfcc(signal, sr=22050, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Process dataset and save features to CSV
features, labels = [], []

for emotion in EMOTIONS:
    folder = os.path.join(DATASET_PATH, emotion)
    
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist!")
        continue
    
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        if file.endswith(".wav"):
            try:
                signal = load_audio(file_path)
                mfcc_features = extract_mfcc(signal)
                features.append(mfcc_features)
                labels.append(emotion)
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(features)
df["label"] = labels

# Save to CSV
os.makedirs(r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\extracted_features", exist_ok=True)
df.to_csv(r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\extracted_features\ser_features.csv", index=False)

print("Feature extraction complete. CSV saved.")

