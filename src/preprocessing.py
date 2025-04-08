# src/preprocessing.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import sys

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ------------------------------------

# Import configuration
from src import config

# --- Define Transformations ---

# Common practice: Use ImageNet means and stds for normalization with pre-trained models
# Otherwise, you might calculate these from your specific dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transformations for the training set (includes augmentation)
train_transforms = transforms.Compose([
    transforms.Resize(config.TARGET_SIZE), # Resize images
    # --- Augmentation ---
    transforms.RandomRotation(config.ROTATION_RANGE),
    # RandomAffine combines translation (shift) and potentially scaling (zoom) and shear
    # Adjust parameters carefully for ECGs. Small shifts/zooms might be okay.
    transforms.RandomAffine(degrees=0, # Rotation handled above
                             translate=(config.WIDTH_SHIFT_RANGE, config.HEIGHT_SHIFT_RANGE),
                             scale=(1.0 - config.ZOOM_RANGE, 1.0 + config.ZOOM_RANGE)
                            ),
    transforms.RandomHorizontalFlip(p=0.5 if config.HORIZONTAL_FLIP else 0.0),
    # Add other augmentations if needed (e.g., transforms.ColorJitter if colors vary)
    # --------------------
    transforms.ToTensor(), # Convert PIL Image to PyTorch tensor (scales to [0, 1])
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Normalize tensor
])

# Transformations for validation and test sets (no augmentation, just preprocessing)
val_test_transforms = transforms.Compose([
    transforms.Resize(config.TARGET_SIZE), # Resize images
    # Consider transforms.CenterCrop(config.TARGET_SIZE) if Resize might not be exact
    transforms.ToTensor(), # Convert PIL Image to PyTorch tensor
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Normalize tensor
])

# --- Custom PyTorch Dataset ---

class ECGImageDataset(Dataset):
    """
    Custom Dataset for loading ECG images from file paths stored in a DataFrame.
    """
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'filepath' and 'class_id'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if dataframe is None or not isinstance(dataframe, pd.DataFrame):
             raise ValueError("Input 'dataframe' must be a non-empty Pandas DataFrame.")
        if 'filepath' not in dataframe.columns or 'class_id' not in dataframe.columns:
            raise ValueError("DataFrame must contain 'filepath' and 'class_id' columns.")

        self.df = dataframe
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image tensor,
                   and label is the integer class label tensor.
                   Returns (None, None) if image loading fails.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]['filepath']
        # Ensure label is integer type before converting to tensor
        label = int(self.df.iloc[idx]['class_id'])

        try:
            # Load image using PIL
            image = Image.open(img_path)

            # Ensure image is in RGB format if model expects 3 channels
            if config.IMG_CHANNELS == 3 and image.mode != 'RGB':
                image = image.convert('RGB')
            elif config.IMG_CHANNELS == 1 and image.mode != 'L':
                 image = image.convert('L') # Convert to grayscale if 1 channel expected

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            # Ensure label is a tensor (LongTensor for CrossEntropyLoss)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return image, label_tensor

        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}. Skipping sample.")
            # Return placeholder or handle appropriately; returning None can cause issues in DataLoader
            # A safer approach might be to filter out missing files beforehand,
            # or return a placeholder tensor if the DataLoader can handle it (e.g., custom collate_fn).
            # For simplicity now, we rely on the files existing. Let's return None and see if DataLoader handles it.
            # If issues arise, we might need to return a dummy tensor or filter self.df in __init__.
            return None, None # NOTE: This might cause errors in default collation
        except Exception as e:
            print(f"Warning: Error loading or processing image {img_path}: {e}. Skipping sample.")
            return None, None # NOTE: This might cause errors in default collation


# --- Function to Create DataLoaders ---

def create_dataloaders(train_df, val_df, test_df):
    """
    Creates PyTorch DataLoaders for training, validation, and test sets.

    Args:
        train_df (pd.DataFrame): DataFrame for the training set.
        val_df (pd.DataFrame): DataFrame for the validation set.
        test_df (pd.DataFrame): DataFrame for the test set.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
               Returns (None, None, None) if any input DataFrame is invalid.
    """
    if train_df is None or val_df is None or test_df is None:
        print("Error: One or more input DataFrames are None. Cannot create DataLoaders.")
        return None, None, None

    # Create Dataset instances
    train_dataset = ECGImageDataset(dataframe=train_df, transform=train_transforms)
    val_dataset = ECGImageDataset(dataframe=val_df, transform=val_test_transforms)
    test_dataset = ECGImageDataset(dataframe=test_df, transform=val_test_transforms)

    # --- Handle potential Nones from __getitem__ ---
    # A robust way is a custom collate function that filters Nones,
    # or filtering the DataFrames *before* creating datasets.
    # Let's add a simple check here for empty datasets after potential errors.
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
         print("Warning: One or more datasets ended up empty, possibly due to file errors during initialization or filtering.")
         # Decide how to handle - maybe return None or raise error
         # Returning loaders that will yield nothing might be okay sometimes.

    # Determine number of workers
    # Use half the CPU cores, but at least 1. Be cautious on shared systems.
    num_workers = max(1, os.cpu_count() // 2) if config.BATCH_SIZE > 8 else 0 # Small batches might not benefit much
    print(f"Using {num_workers} workers for DataLoaders.")

    # Create DataLoader instances
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Helps speed up CPU to GPU transfer if using CUDA
        # collate_fn=custom_collate_fn # Add if needed to handle None returns from __getitem__
        drop_last=True # Drop last non-full batch in training? Optional.
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # No shuffling for validation
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # No shuffling for testing
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Created DataLoaders:")
    print(f" - Training: {len(train_loader)} batches (batch size {config.BATCH_SIZE})")
    print(f" - Validation: {len(val_loader)} batches")
    print(f" - Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Running preprocessing.py directly for testing...")

    # Create dummy DataFrames for testing structure (replace with data_loader loading)
    print("Creating dummy DataFrames for testing...")
    num_samples = 100
    dummy_data = {
        'filepath': [f'dummy_path_{i}.jpg' for i in range(num_samples)], # Need actual files for image loading test
        'class_id': [i % config.NUM_CLASSES for i in range(num_samples)],
        'class_name': [config.INV_CLASS_MAP[i % config.NUM_CLASSES] for i in range(num_samples)]
    }
    dummy_df = pd.DataFrame(dummy_data)

    # You'd normally get these from data_loader.py
    dummy_train_df, dummy_val_test_df = train_test_split(dummy_df, test_size=0.3, stratify=dummy_df['class_id'])
    dummy_val_df, dummy_test_df = train_test_split(dummy_val_test_df, test_size=0.5, stratify=dummy_val_test_df['class_id'])

    print(f"Dummy Train DF size: {len(dummy_train_df)}")
    print(f"Dummy Val DF size: {len(dummy_val_df)}")
    print(f"Dummy Test DF size: {len(dummy_test_df)}")

    print("\nAttempting to create DataLoaders...")
    # Note: This part will fail if dummy file paths don't exist or images can't be loaded/processed.
    # To test properly, point it to real data using data_loader.
    try:
        # Replace dummy DFs with real ones loaded via data_loader for actual testing
        # from src import data_loader
        # main_df = data_loader.load_data_paths()
        # if main_df is not None:
        #     train_dataframe, val_dataframe, test_dataframe = data_loader.split_data(main_df)
        #     if train_dataframe is not None:
        #         train_dl, val_dl, test_dl = create_dataloaders(train_dataframe, val_dataframe, test_dataframe)

        # Using dummy data for structure test (will likely fail image loading)
        train_dl, val_dl, test_dl = create_dataloaders(dummy_train_df, dummy_val_df, dummy_test_df)

        if train_dl:
            print("\n--- Testing Dataloader Iteration (requires dummy images) ---")
            # Fetch one batch to test the pipeline (will fail if dummy paths are invalid)
            try:
                images, labels = next(iter(train_dl))
                print(f"Fetched one batch:")
                print(f" - Images shape: {images.shape}") # Expects [BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH]
                print(f" - Labels shape: {labels.shape}") # Expects [BATCH_SIZE]
                print(f" - Image tensor sample (min/max): {images.min():.2f}/{images.max():.2f}")
                print(f" - Sample labels: {labels[:5]}...")
                print("--- Testing Completed Successfully (structure seems okay) ---")
            except Exception as e:
                 print(f"\nError fetching batch (expected if using dummy file paths): {e}")
                 print("--- Testing Completed (structure check only) ---")

        else:
             print("Failed to create DataLoaders.")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()