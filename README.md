This project implements a multimodal neural network that combines:
- Image embeddings from a custom Keras-serializable CLIPImageEncoder
- Tabular numerical features

To create a fused network that outputs a binary classification of NFL play types (run/pass play)

It uses TensorFlow/Keras, Open AI's CLIP, and a custom architecture that can be trained and tested separately.

Features:
- Custom CLIP Image Encoder using Hugging Face's openai/clip-vit-base-patch32
- @keras.saving.register_keras_serializable support for full model serialization
- Multimodal model combining image + numeric features
- Separate training (train_multimodal.py) and testing (test.py) files
- Support for evaluating accuracy over multiple test samples
- Virtual environment recommended (Python 3.11)

Files:

train_multimodal.py       # Builds and trains the multimodal model

test.py                   # Loads the saved model and runs inference

Model Overview:
1) CLIPImageEncoder (Custom Layer)

A serializable Keras layer wrapping the CLIP image encoder:
- Registered using @keras. saving register_keras_serializable
- Imports CLIP via Hugging Face Transformers
- Produces a 512-dim image embedding

2) Numeric Feature Encoder
- Standardized numerical features
- Fully connected Dense layers

3) Fusion Layer
- Concatenates image + numeric embeddings
- Outputs a binary classification prediction

Training the Model:

Run:

python3 train_multimodal.py

This will:
- Build the model
- Train on your image + numeric dataset
- Save the model multimodal_clip_model. keras

Testing:

* Make sure your virtual environment is active:

source venv/bin/activate

Run:

python3 test.py

The test script:

- Imports CLIPImageEncoder
- Loads the saved Keras model
- Loops over test samples
- Computes accuracy
- Prints the predictions


Requirements/Setup Instructions:
1. Create a virtual environment:

python -m venv venv source venv/bin/activate

2. Install dependencies:

tensorflow==2.17.0

transformers pillow numpy pandas scikit-learn keras
