import os

# --- Data Paths ---
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This gets the 'ecg_classification' directory path

# Raw Data Path (Update if your structure is different)
# Assumes data is in ./data/raw/ClassName/image.jpg relative to project root
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Saved Model Path
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
MODEL_NAME = 'ecg_cnn_standard_v1.keras' # Keras 3 recommended format
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)


# --- Image Parameters ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3 # Assuming RGB images. If grayscale, change to 1.
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
TARGET_SIZE = (IMG_HEIGHT, IMG_WIDTH) # Required for ImageDataGenerator

# --- Dataset Parameters ---
NUM_CLASSES = 4 # MI, Abnormal Heartbeat, History MI, Normal
# Define class names - IMPORTANT: Ensure this order matches the folder names
# alphabetically if relying on os.listdir, or define explicitly based on mapping.
# Using sorted listdir output is safer if folders exist.
# We will confirm/set this properly in data_loader.py
CLASS_NAMES = ['Myocardial Infarction', 'Abnormal Heartbeat', 'History of MI', 'Normal'] # Example - verify this matches your folders

# --- Training Parameters ---
BATCH_SIZE = 32 # Adjust based on your GPU memory
EPOCHS = 50 # Start with a reasonable number, adjust later
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15 # 15% of the data for validation
TEST_SPLIT = 0.15 # 15% of the data for testing

# --- Data Augmentation Parameters (for training generator) ---
ROTATION_RANGE = 10       # degrees
WIDTH_SHIFT_RANGE = 0.1   # fraction of total width
HEIGHT_SHIFT_RANGE = 0.1  # fraction of total height
ZOOM_RANGE = 0.1          # zoom factor range [1-zoom_range, 1+zoom_range]
HORIZONTAL_FLIP = True    # Allow horizontal flipping
VERTICAL_FLIP = False     # Usually False for ECGs
FILL_MODE = 'nearest'

# --- Miscellaneous ---
RANDOM_STATE = 42 # For reproducibility in splits

# --- Derived Configuration ---
# Create class map dictionaries dynamically (safer)
if CLASS_NAMES:
    CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
    INV_CLASS_MAP = {i: name for name, i in CLASS_MAP.items()}
else:
    # Handle case where class names might be derived later in data_loader
    CLASS_MAP = {}
    INV_CLASS_MAP = {}


# Print configuration sanity check when module is loaded (optional)
# print("--- Configuration Loaded ---")
# print(f"Project Base Directory: {BASE_DIR}")
# print(f"Raw Data Directory: {RAW_DATA_DIR}")
# print(f"Model Save Path: {MODEL_SAVE_PATH}")
# print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
# print(f"Batch Size: {BATCH_SIZE}")
# print(f"Number of Classes: {NUM_CLASSES}")
# print(f"Class Names: {CLASS_NAMES}")
# print("--------------------------")
