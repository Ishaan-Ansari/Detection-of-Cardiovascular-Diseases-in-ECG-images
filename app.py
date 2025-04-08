# app.py

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
import io # To handle byte stream from file uploader
import pandas as pd

# --- Add project root to sys.path ---
# This allows importing 'src' modules when running 'streamlit run app.py' from the root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------

# Import project modules after updating path
from src import config
from src import model as model_builder
from src import preprocessing # Import to access transforms

# --- Configuration ---
MODEL_PATH = config.MODEL_SAVE_PATH
NUM_CLASSES = config.NUM_CLASSES
CLASS_NAMES = config.CLASS_NAMES # Assumes config has the final list
IMG_CHANNELS = config.IMG_CHANNELS
TARGET_SIZE = config.TARGET_SIZE

# Check if CLASS_NAMES are available
if not CLASS_NAMES or len(CLASS_NAMES) != NUM_CLASSES:
    st.error(f"Error: Class names configuration issue. Expected {NUM_CLASSES} names.")
    # Attempt to load dynamically (might require running data_loader logic, complex here)
    # For simplicity, ensure config.py reflects the classes trained on.
    # If data_loader updated config dynamically, restarting the app *might* pick it up
    # A more robust solution involves saving class names with the model or in a separate file.
    st.stop() # Stop execution if class names are not properly configured

# Use the validation/test transforms defined in preprocessing.py
# Make sure preprocessing.py defines these correctly
try:
    inference_transforms = preprocessing.val_test_transforms
except AttributeError:
    st.error("Error: Could not find 'val_test_transforms' in src.preprocessing.py.")
    st.stop()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model (Cached) ---
@st.cache_resource # Use cache_resource for non-data objects like models
def load_pytorch_model(model_path, num_classes, device):
    """Loads the PyTorch model from the specified path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Instantiate model architecture (same as during training/evaluation)
    # Assuming freeze_feature_extractor=True was used for saving the relevant state_dict
    model = model_builder.create_model(num_classes=num_classes, freeze_feature_extractor=True)

    # Load the saved state dictionary
    try:
        # Use map_location for flexibility between CPU/GPU
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise IOError(f"Error loading model weights: {e}")

    model = model.to(device)
    model.eval() # Set to evaluation mode
    return model

# --- Prediction Function ---
def predict(model, pil_image, transform, device, class_names_list):
    """Makes a prediction on a PIL image."""
    # Ensure image is RGB if model expects 3 channels
    if IMG_CHANNELS == 3 and pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    elif IMG_CHANNELS == 1 and pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    # Apply transformations
    image_tensor = transform(pil_image).unsqueeze(0) # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names_list[predicted_idx.item()]
    confidence_score = confidence.item()

    # Get probabilities for all classes
    all_probs = probabilities.cpu().numpy().flatten()
    prob_dict = {class_names_list[i]: all_probs[i] for i in range(len(class_names_list))}

    return predicted_class, confidence_score, prob_dict


# --- Streamlit App UI ---
st.set_page_config(page_title="ECG Image Classification", layout="wide")

st.title("🩺 ECG Image Classification")
st.write("Upload an ECG image (JPG, PNG, BMP) to classify it into one of the categories:")
st.write(f"**Categories:** {', '.join(CLASS_NAMES)}")
st.markdown("---") # Separator

# Load the model (will be cached after first run)
try:
    model = load_pytorch_model(MODEL_PATH, NUM_CLASSES, device)
    st.sidebar.success(f"Model loaded successfully from {MODEL_PATH}")
    st.sidebar.write(f"Running on device: {device}")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop() # Stop the app if model loading fails

# File Uploader
uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Read the file bytes
    bytes_data = uploaded_file.getvalue()
    # Open image using PIL directly from bytes
    try:
        image = Image.open(io.BytesIO(bytes_data))

        # Display the uploaded image
        st.image(image, caption='Uploaded ECG Image.', use_column_width=True)
        st.markdown("---")

        # Make prediction when image is uploaded
        st.write("Classifying...")
        with st.spinner('Model is predicting...'):
            predicted_class, confidence_score, all_probabilities = predict(model, image, inference_transforms, device, CLASS_NAMES)

        st.success(f"Prediction Complete!")
        st.subheader("Results:")
        st.write(f"Predicted Condition: **{predicted_class}**")
        st.write(f"Confidence: **{confidence_score:.4f}**")

        # Display probabilities for all classes
        st.subheader("Class Probabilities:")
        # Convert dict to DataFrame for better display/charting
        prob_df = pd.DataFrame(list(all_probabilities.items()), columns=['Class', 'Probability'])
        st.dataframe(prob_df.style.format({'Probability': "{:.4f}"}))
        st.bar_chart(prob_df.set_index('Class'))


    except Exception as e:
        st.error(f"Error processing image or predicting: {e}")
        st.error("Please ensure the uploaded file is a valid image.")

else:
    st.info("Please upload an image file to get a prediction.")

st.markdown("---")
st.sidebar.markdown("## About")
st.sidebar.info("This app uses a ResNet50 model trained to classify ECG images.")