# src/data_loader.py

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import sys # To modify sys.path for importing config

# --- Add project root to sys.path ---
# This allows importing 'src.config' even when running scripts from nested directories
# Adjust the number of '..' based on where you run your scripts from.
# If running 'train.py' from the project root 'ecg_classification/', this might not be strictly needed
# but it's safer for potentially running scripts from elsewhere or within an IDE.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ------------------------------------

# Import configuration after potentially modifying sys.path
from src import config # Import the config module


def load_data_paths():
    """
    Scans the raw data directory specified in config, identifies class folders,
    finds image file paths, assigns labels, and returns a Pandas DataFrame.
    Updates config.CLASS_NAMES and related variables based on found folders.
    """
    raw_data_dir = config.RAW_DATA_DIR
    image_paths = []
    labels = []
    class_names_found_list = []

    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw data directory not found at {raw_data_dir}")
        return None # Indicate failure

    print(f"Scanning directory: {raw_data_dir}")
    # Find directories (potential classes), sort them for consistent order
    try:
        class_folders = sorted([f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))])
    except FileNotFoundError:
        print(f"Error: Raw data directory not found or inaccessible: {raw_data_dir}")
        return None

    if not class_folders:
        print(f"Error: No subdirectories (classes) found in {raw_data_dir}")
        return None

    print(f"Found class folders: {class_folders}")

    # --- Dynamically update config based on found folders ---
    # This overrides the initial CLASS_NAMES in config.py, which is safer
    config.CLASS_NAMES = class_folders
    config.NUM_CLASSES = len(class_folders)
    config.CLASS_MAP = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    config.INV_CLASS_MAP = {i: name for name, i in config.CLASS_MAP.items()}
    print(f"Updated config CLASS_NAMES to: {config.CLASS_NAMES}")
    print(f"Updated config NUM_CLASSES to: {config.NUM_CLASSES}")
    # ---------------------------------------------------------

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(raw_data_dir, class_name)
        class_label = config.CLASS_MAP[class_name]

        # Use glob to find common image file types
        # Consider adding more extensions if needed (e.g., '.tif')
        files_in_class = glob.glob(os.path.join(class_dir, '*.jpg')) \
                       + glob.glob(os.path.join(class_dir, '*.png')) \
                       + glob.glob(os.path.join(class_dir, '*.jpeg')) \
                       + glob.glob(os.path.join(class_dir, '*.bmp'))

        for file_path in files_in_class:
            image_paths.append(file_path)
            labels.append(class_label)
            class_names_found_list.append(class_name) # Store the string name too

    if not image_paths:
         print("Error: No images found. Please check the directory structure and image file extensions.")
         return None

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        'filepath': image_paths,
        'class_id': labels, # Integer label
        'class_name': class_names_found_list # String label
    })

    # --- Add Patient ID column (Placeholder) ---
    # This is where you would extract the patient ID if possible, e.g., from filename
    # Example (assuming filename format 'PatientID_***.jpg'):
    # try:
    #     df['patient_id'] = df['filepath'].apply(lambda x: os.path.basename(x).split('_')[0])
    # except Exception as e:
    #     print(f"Warning: Could not automatically extract patient IDs from filenames ({e}). Proceeding without patient IDs.")
    #     df['patient_id'] = None # Indicate missing patient info
    #
    # If patient IDs are available, the split_data function should be modified for patient-aware splitting.
    # For now, we proceed without it.
    print("Note: Patient ID extraction not implemented by default. Modify 'load_data_paths' if possible.")
    # -------------------------------------------


    print(f"\nLoaded {len(df)} images belonging to {config.NUM_CLASSES} classes.")
    print("\nSample of the DataFrame:")
    print(df.head())
    print("\nClass Distribution:")
    print(df['class_name'].value_counts())

    return df


def split_data(df):
    """
    Splits the DataFrame into training, validation, and test sets using stratified splitting.

    Args:
        df (pd.DataFrame): DataFrame containing 'filepath', 'class_id', 'class_name'.

    Returns:
        tuple: (train_df, val_df, test_df)
               Returns (None, None, None) if input df is invalid.
    """
    if df is None or df.empty:
        print("Error: Input DataFrame is empty or None. Cannot split data.")
        return None, None, None

    test_split_size = config.TEST_SPLIT
    val_split_size = config.VALIDATION_SPLIT
    random_state = config.RANDOM_STATE

    # Calculate the size of the training set needed after removing test and validation
    train_split_size = 1.0 - test_split_size - val_split_size
    if train_split_size <= 0 or test_split_size <=0 or val_split_size <=0:
         print(f"Error: Invalid split ratios. Train ({train_split_size}), Val ({val_split_size}), Test ({test_split_size}). Sum must be < 1 and all > 0.")
         return None, None, None

    # First split: Separate Test set
    # test_size here is the fraction of the original dataset
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_split_size,
        random_state=random_state,
        stratify=df['class_id'] # Ensure class proportions are similar
    )

    # Second split: Split remaining data into Training and Validation
    # The validation size here needs to be relative to the size of train_val_df
    # val_size_relative = val_split_size / (train_split_size + val_split_size)
    val_size_relative = val_split_size / (1.0 - test_split_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_relative,
        random_state=random_state, # Use same random state for consistency in relative split
        stratify=train_val_df['class_id'] # Stratify again
    )

    print("\n--- Data Splitting ---")
    print(f"Total images: {len(df)}")
    print(f"Training set size: {len(train_df)} images ({len(train_df)/len(df):.1%})")
    print(f"Validation set size: {len(val_df)} images ({len(val_df)/len(df):.1%})")
    print(f"Test set size: {len(test_df)} images ({len(test_df)/len(df):.1%})")

    # Optional: Check class distribution in each split
    print("\nTraining Set Class Distribution:")
    print(train_df['class_name'].value_counts(normalize=True).sort_index())
    print("\nValidation Set Class Distribution:")
    print(val_df['class_name'].value_counts(normalize=True).sort_index())
    print("\nTest Set Class Distribution:")
    print(test_df['class_name'].value_counts(normalize=True).sort_index())

    print("\nIMPORTANT NOTE: This is a standard stratified random split.")
    print("      For datasets with multiple samples per subject (like ECGs per patient),")
    print("      a patient-aware split is STRONGLY recommended to avoid data leakage")
    print("      and obtain a more realistic performance estimate. This requires")
    print("      implementing patient ID extraction in 'load_data_paths' and")
    print("      modifying the 'split_data' logic to split based on unique patient IDs.")

    return train_df, val_df, test_df


# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Running data_loader.py directly for testing...")

    # Load data paths into DataFrame
    main_df = load_data_paths()

    if main_df is not None:
        # Split the data
        train_dataframe, val_dataframe, test_dataframe = split_data(main_df)

        if train_dataframe is not None:
            print("\n--- Testing Completed ---")
            print("Sample of Training DataFrame head:")
            print(train_dataframe.head())
            print("\nSample of Validation DataFrame head:")
            print(val_dataframe.head())
            print("\nSample of Test DataFrame head:")
            print(test_dataframe.head())
        else:
            print("Splitting failed.")
    else:
        print("Loading data paths failed.")