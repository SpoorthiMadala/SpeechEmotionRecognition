import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load extracted features
csv_path = r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\extracted_features\ser_features.csv"
df = pd.read_csv(csv_path)

print("🔍 Checking dataset...")
print(df.head())  # Show first few rows
print("Dataset shape:", df.shape)

# Check if 'label' column exists
if "label" not in df.columns:
    raise ValueError("❌ Error: 'label' column missing in ser_features.csv!")

# Separate features (X) and labels (y)
X = df.iloc[:, :-1].values  # Features
y = df["label"].values      # Labels

print("✅ Features shape (X):", X.shape)
print("✅ Labels shape (y):", y.shape)

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split (same split used in training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# If test set is empty, stop
if X_test.shape[0] == 0:
    raise ValueError("❌ Error: Test set is empty! Check dataset.")

# Load Model
model_path = r"C:\Users\venkatesh\OneDrive - vitap.ac.in\Documents\SER_Project\models\speech_emotion_ann.h5"
model = tf.keras.models.load_model(model_path)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

