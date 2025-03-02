import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Load Model
model_path = r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\models\speech_emotion_ann.h5"
model = tf.keras.models.load_model(model_path)

# Load Labels
csv_path = r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\extracted_features\ser_features.csv"
df = pd.read_csv(csv_path)
label_encoder = LabelEncoder()
label_encoder.fit(df["label"].values)

# Function to extract MFCC
def extract_mfcc(file_path, sr=22050, n_mfcc=40):
    signal, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Predict Emotion
def predict_emotion(file_path):
    if not os.path.isfile(file_path):
        print("Error: File not found!")
        return None
    
    features = extract_mfcc(file_path).reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_label[0]

# Get User Input for File Path
if __name__ == "__main__":
    file_path = input("Enter the path to the audio file: ").strip()
    emotion = predict_emotion(file_path)
    if emotion:
        print(f"Predicted Emotion: {emotion}")

