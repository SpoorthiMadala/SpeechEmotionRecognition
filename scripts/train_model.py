import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load extracted features
csv_path = r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\extracted_features\ser_features.csv"
df = pd.read_csv(csv_path)

print("Checking dataset...")
print(df.head())  # Show first few rows
print("Dataset shape:", df.shape)

# Check if the 'label' column exists
if "label" not in df.columns:
    raise ValueError("Error: 'label' column missing in ser_features.csv!")

# Separate features (X) and labels (y)
X = df.iloc[:, :-1].values  # Features (all columns except last)
y = df["label"].values      # Labels (last column)

print("Features shape (X):", X.shape)
print("Labels shape (y):", y.shape)

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# If train set is empty, stop
if X_train.shape[0] == 0:
    raise ValueError("Error: Training set is empty! Check dataset.")

# Build ANN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(set(y)), activation="softmax")
])

# Compile Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save(r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\models\speech_emotion_ann.h5")
print("Model training complete. Model saved.")

