import tensorflow as tf
from transformers import CLIPProcessor, TFCLIPModel
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
from train_multimodal import CLIPImageEncoder

# ==========================================================
# 2. Load your trained model
# ==========================================================
print("Loading multimodal_clip_model.keras...")
model = keras.models.load_model(
    "multimodal_clip_model.keras",
    custom_objects={"CLIPImageEncoder": CLIPImageEncoder},
    compile=False
)
print("Model loaded successfully.\n")


# ==========================================================
# 3. Load a real CSV row + preprocess the tabular data
# ==========================================================
df = pd.read_csv("data/test_data.csv")

# Tabular columns
tab_cols = ["yd_line", "down", "yds_to_go", "score_diff", "quarter", "time_rem"]

# Convert MM:SS → seconds
def time_to_seconds(t):
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except:
        return 0

df["time_rem"] = df["time_rem"].apply(time_to_seconds)

# Fit scaler on entire CSV like training did
scaler = StandardScaler()
df[tab_cols] = scaler.fit_transform(df[tab_cols])

# Select ONE row to test on (change index if you want)
row = df.iloc[56]
print("Using this CSV row:")
print(row)

image_filename = row["filename"]
image_path = os.path.join("data/test_images", image_filename)

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")


# ==========================================================
# 4. Process the image through CLIP processor
# ==========================================================
print("\nLoading image:", image_path)
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))
image = np.array(image).astype("float32")  # raw 0–255
pixel_values = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)


# ==========================================================
# 5. Format tabular input
# ==========================================================
tab_vector = row[tab_cols].values.astype("float32")
tab_vector = np.expand_dims(tab_vector, axis=0)  # (1, num_features)


# ==========================================================
# 6. Run model prediction
# ==========================================================
prediction = model.predict({"image": pixel_values, "tabular": tab_vector})
prob = float(prediction[0][0])

label = "PASS" if prob >= 0.65 else "RUN"

# ==========================================================
# 7. Print results
# ==========================================================
print("\n========================")
print("MODEL PREDICTION RESULTS")
print("========================")
print(f"Predicted probability of PASS: {prob:.4f}")
print(f"Predicted label: {label}")
