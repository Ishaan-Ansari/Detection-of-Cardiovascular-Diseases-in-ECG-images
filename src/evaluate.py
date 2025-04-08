# src/evaluate.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import time
from tqdm import tqdm
import sys

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ------------------------------------

# Import project modules
from src import config
from src import data_loader
from src import preprocessing
from src import model as model_builder

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plots a confusion matrix using Seaborn heatmap.

    Args:
        cm (np.ndarray): Confusion matrix array.
        class_names (list): List of class names for axis labels.
        save_path (str, optional): Path to save the plot image. Defaults to None (display only).
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        if save_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to: {save_path}")
            plt.close() # Close the plot if saving to file
        else:
            plt.show() # Display the plot if not saving
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")


def evaluate_model():
    """
    Loads the trained model and evaluates it on the test set.
    """
    print("--- Starting Model Evaluation ---")
    start_time = time.time()

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data (to get test set and ensure consistent class mapping) ---
    print("Loading data paths and ensuring class consistency...")
    main_df = data_loader.load_data_paths()
    if main_df is None or main_df.empty:
        print("Error: Failed to load data paths. Exiting.")
        return

    # Ensure class names in config are up-to-date based on loaded data
    # This is crucial if data_loader dynamically determines classes
    if not config.CLASS_NAMES or len(config.CLASS_NAMES) != config.NUM_CLASSES:
         print("Error: Class names/number mismatch after loading data. Check config and data_loader.")
         return

    print("Splitting data to isolate test set...")
    train_df, val_df, test_df = data_loader.split_data(main_df)
    if test_df is None:
        print("Error: Failed to get test data split. Exiting.")
        return

    # --- Create Test DataLoader ---
    print("Creating Test DataLoader...")
    # We only need the test_loader from this function call
    _, _, test_loader = preprocessing.create_dataloaders(train_df, val_df, test_df)
    if test_loader is None:
        print("Error: Failed to create Test DataLoader. Exiting.")
        return

    # --- Load Model ---
    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print(f"Error: Saved model not found at {model_path}. Run train.py first.")
        return

    print(f"Loading model from: {model_path}")
    # Create model architecture - use same settings as during training for loading state_dict
    # E.g., if trained with freeze_feature_extractor=True, create it that way here too.
    # PyTorch load_state_dict is flexible, but matching architecture avoids potential issues.
    # Let's assume it was trained with freeze=True as per train.py default.
    model = model_builder.create_model(num_classes=config.NUM_CLASSES, freeze_feature_extractor=True)

    try:
        # Load the saved weights. map_location ensures model loads correctly even if
        # trained on GPU and evaluated on CPU, or vice-versa.
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    model = model.to(device) # Move model to the evaluation device
    model.eval() # Set model to evaluation mode (important for layers like Dropout, BatchNorm)

    # --- Evaluation Loop ---
    print("\nRunning evaluation on the test set...")
    all_labels = []
    all_preds = []

    with torch.no_grad(): # Disable gradient calculations for inference
        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, labels in pbar:
            if inputs is None or labels is None: # Basic check
                continue

            inputs = inputs.to(device)
            labels = labels.to(device) # Keep labels on device for potential comparison

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Collect labels and predictions (move to CPU for numpy/sklearn)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # --- Calculate Metrics ---
    if not all_labels:
        print("Error: No predictions were made. Check the test loader and evaluation loop.")
        return

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print("\n--- Evaluation Results ---")

    # Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Test Accuracy: {accuracy:.4f}")

    # Classification Report (Precision, Recall, F1-score per class)
    print("\nClassification Report:")
    # Ensure class names from config are correct and match the labels' order
    if len(config.CLASS_NAMES) == config.NUM_CLASSES:
         class_report = classification_report(all_labels, all_preds,
                                             target_names=config.CLASS_NAMES, digits=4)
         print(class_report)
    else:
         print("Warning: Number of class names in config doesn't match NUM_CLASSES. Cannot generate report with names.")
         # Generate report without names
         class_report = classification_report(all_labels, all_preds, digits=4)
         print(class_report)


    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    # Format the matrix nicely for printing
    cm_df = pd.DataFrame(cm, index=config.CLASS_NAMES, columns=config.CLASS_NAMES)
    print(cm_df)

    # --- Plot Confusion Matrix ---
    cm_plot_path = os.path.join(config.BASE_DIR, 'reports', 'figures', 'confusion_matrix.png')
    plot_confusion_matrix(cm, config.CLASS_NAMES, save_path=cm_plot_path)


    # --- Evaluation Complete ---
    eval_time = time.time() - start_time
    print(f'\n--- Evaluation complete in {eval_time // 60:.0f}m {eval_time % 60:.0f}s ---')


# --- Script Execution ---
if __name__ == '__main__':
    evaluate_model()