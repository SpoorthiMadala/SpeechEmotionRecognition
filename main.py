import os

print("🚀 Starting Speech Emotion Recognition Project...\n")

# Step 1: Extract Features
print("📂 Step 1: Extracting Features...")
if os.system("python scripts/data_preprocessing.py") == 0:
    print("✅ Feature Extraction Completed.\n")
else:
    print("❌ Feature Extraction Failed. Check for errors!")
    exit()

# Step 2: Train Model
print("🎯 Step 2: Training Model...")
if os.system("python scripts/train_model.py") == 0:
    print("✅ Model Training Completed.\n")
else:
    print("❌ Model Training Failed. Check for errors!")
    exit()

# Step 3: Evaluate Model
print("📊 Step 3: Evaluating Model...")
if os.system("python scripts/evaluate_model.py") == 0:
    print("✅ Model Evaluation Completed.\n")
else:
    print("❌ Model Evaluation Failed. Check for errors!")
    exit()

print("📊 Step 4: Predicting Model...")
if os.system("python scripts/predict_emotion.py") == 0:
    print("✅ Model Prediction Completed.\n")
else:
    print("❌ Model Prediction Failed. Check for errors!")
    exit()


print("🎉 SER Project Completed Successfully! 🚀")

