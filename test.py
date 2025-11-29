import tensorflow as tf
from transformers import CLIPProcessor, TFCLIPModel
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
from train_multimodal import CLIPImageEncoder


# 1. Load trained model
print("Loading multimodal_clip_model.keras...")
model = keras.models.load_model(
    "multimodal_clip_model.keras",
    custom_objects={"CLIPImageEncoder": CLIPImageEncoder},
    compile=False
)
print("Model loaded successfully.\n")


# 2. Load CSV
df = pd.read_csv("data/test_data.csv")

# Tabular columns
tab_cols = ["yd_line", "down", "yds_to_go", "score_diff", "quarter", "time_rem"]

# Convert time from MM:SS to seconds
def time_to_seconds(t):
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except:
        return 0

df["time_rem"] = df["time_rem"].apply(time_to_seconds)

# Scale tabular features (same as training)
scaler = StandardScaler()
df[tab_cols] = scaler.fit_transform(df[tab_cols])


#  Looping through test samples
print("\nRunning predictions on all test samples...\n")

results = []
correct = 0
total = 0

for idx, row in df.iterrows():
    image_filename = row["filename"]
    image_path = os.path.join("data/test_images", image_filename)

    if not os.path.exists(image_path):
        print(f"[WARNING] Image not found: {image_path}")
        continue

    # Ground truth label
    true_label = row["label"].upper()  # assumes PASS or RUN

    # Image preprocessing
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype("float32")
    pixel_values = np.expand_dims(image, axis=0)

    # Tabular preprocessing
    tab_vector = row[tab_cols].values.astype("float32")
    tab_vector = np.expand_dims(tab_vector, axis=0)

    # Run prediction
    prediction = model.predict({"image": pixel_values, "tabular": tab_vector}, verbose=0)
    prob = float(prediction[0][0])
    pred_label = "PASS" if prob >= 0.65 else "RUN"

    # Compare prediction vs truth
    is_correct = (pred_label == true_label)
    correct += int(is_correct)
    total += 1

    results.append((idx, image_filename, prob, pred_label, true_label, is_correct))

    print(f"Row {idx:02d} | {image_filename} | PASS prob: {prob:.4f} | Pred: {pred_label} | True: {true_label} | {'✓' if is_correct else '✗'}")


#  Summary printout
accuracy = correct / total if total > 0 else 0

print("\n========================")
print("      TEST RESULTS")
print("========================")
for idx, filename, prob, pred_label, true_label, is_correct in results:
    print(f"{idx:02d}: {filename:25s}  Prob={prob:.4f}  Pred={pred_label}  True={true_label}  {'✓' if is_correct else '✗'}")

print("\n========================")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy*100:.2f}%")
print("========================")
