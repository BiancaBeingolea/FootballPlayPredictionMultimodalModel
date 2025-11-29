import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, TFCLIPModel
from tensorflow.keras.utils import Sequence
from keras.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os

# 1. Loading the dataset (returns raw 224×224 RGB pixels)
class PlayDataset(Sequence):
    def __init__(self, df, image_dir, scaler, batch_size=16):
        self.df = df
        self.image_dir = image_dir
        self.scaler = scaler
        self.batch_size = batch_size

        self.tab_cols = ["yd_line", "down", "yds_to_go",
                         "score_diff", "quarter", "time_rem"]

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]

        images = []
        tabular = []
        labels = []

        for _, row in batch.iterrows():
            # load raw image
            img_path = os.path.join(self.image_dir, row["filename"])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img = np.array(img).astype("float32")  # raw 0–255
            images.append(img)

            #  handle tabular data
            tabular.append(row[self.tab_cols].values.astype("float32"))

            # add label
            labels.append(1 if row["label"] == "pass" else 0)

        return {
            "image": tf.convert_to_tensor(np.stack(images), dtype=tf.float32),
            "tabular": tf.convert_to_tensor(self.scaler.transform(tabular), dtype=tf.float32)
        }, tf.convert_to_tensor(labels, dtype=tf.float32)


# 2. Incorporating CLIP Image Encoder for embeddings
@register_keras_serializable(name="CLIPImageEncoder")
class CLIPImageEncoder(tf.keras.layers.Layer):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", **kwargs):
        super().__init__(**kwargs)

        # serializable settings
        self.clip_model_name = clip_model_name

        # Load HuggingFace model + processor
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = TFCLIPModel.from_pretrained(clip_model_name)

    def _encode_numpy(self, images_np):
        """
        Runs CLIP in eager mode (via numpy → python).
        This is required because HuggingFace internals
        cannot be traced by TensorFlow graph.
        """
        imgs = list(images_np)  # numpy to list of images

        processed = self.processor(images=imgs, return_tensors="tf")
        outputs = self.model.get_image_features(**processed)
        return outputs.numpy()  # returning numpy for tf.py_function

    def call(self, inputs):
        outputs = tf.py_function(
            func=self._encode_numpy,
            inp=[inputs],
            Tout=tf.float32
        )

        outputs.set_shape((None, 512)) # 512-dim embeddings
        return outputs

    def get_config(self):
        config = super().get_config()
        config["clip_model_name"] = self.clip_model_name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# 3. Building Multimodal Model
def build_multimodal_model(num_tab_features):
    image_input = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32, name="image")
    tabular_input = tf.keras.Input(shape=(num_tab_features,), dtype=tf.float32, name="tabular")

    image_encoder = CLIPImageEncoder()
    clip_features = image_encoder(image_input)

    x = tf.keras.layers.Concatenate()([clip_features, tabular_input])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[image_input, tabular_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# 4. Loading CSV & Preprocessing Tabular Data
df = pd.read_csv("data/metadata.csv")

tabular_cols = ["yd_line", "down", "yds_to_go",
                "score_diff", "quarter", "time_rem"]

# Converting time from MM:SS to integer
def time_to_seconds(t):
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except:
        return 0

df["time_rem"] = df["time_rem"].apply(time_to_seconds)

# Scaling tabular values
scaler = StandardScaler()
df[tabular_cols] = scaler.fit_transform(df[tabular_cols])

# Creating train/validation split
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

train_data = PlayDataset(train_df, "data/images", scaler)
val_data = PlayDataset(val_df, "data/images", scaler)


# 5. Training & Saving Model
if __name__ == "__main__":
    model = build_multimodal_model(num_tab_features=len(tabular_cols))
    model.summary()

    history = model.fit(train_data, validation_data=val_data, epochs=10)

    model.save("multimodal_clip_model.keras")
    print("Model saved to multimodal_clip_model.keras")
