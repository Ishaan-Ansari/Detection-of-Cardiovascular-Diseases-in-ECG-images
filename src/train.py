# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler # Optional: for learning rate scheduling
import os
import time
import copy # To save the best model weights
from tqdm import tqdm # For nice progress bars
import sys
import pandas as pd # For checking DataFrame validity

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ------------------------------------

# Import project modules
from src import config
from src import data_loader
from src import preprocessing
from src import model as model_builder # Use alias to avoid naming conflict

def train_model():
    """
    Main function to orchestrate the model training process.
    """
    print("--- Starting Model Training ---")
    start_time = time.time()

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading data paths...")
    main_df = data_loader.load_data_paths()
    if main_df is None or main_df.empty:
        print("Error: Failed to load data paths. Exiting.")
        return

    print("Splitting data...")
    train_df, val_df, test_df = data_loader.split_data(main_df)
    if train_df is None or val_df is None or test_df is None:
        print("Error: Failed to split data. Exiting.")
        return

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    train_loader, val_loader, _ = preprocessing.create_dataloaders(train_df, val_df, test_df)
    # We don't need the test_loader during training itself
    if train_loader is None or val_loader is None:
        print("Error: Failed to create DataLoaders. Exiting.")
        return

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_df), 'val': len(val_df)}
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Validation={dataset_sizes['val']}")


    # --- Create Model ---
    print("Creating model...")
    # Ensure config.NUM_CLASSES is updated by data_loader if it was dynamic
    if not hasattr(config, 'NUM_CLASSES') or config.NUM_CLASSES <= 0:
        print("Error: Number of classes not properly configured. Exiting.")
        return

    model = model_builder.create_model(num_classes=config.NUM_CLASSES, freeze_feature_extractor=True)
    model = model.to(device) # Move model to GPU or CPU

    # --- Define Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    print(f"Using Loss Function: CrossEntropyLoss")

    # Observe that only parameters of the final layer are being optimized
    # if freeze_feature_extractor=True in create_model
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=config.LEARNING_RATE)
    print(f"Using Optimizer: Adam (LR={config.LEARNING_RATE})")

    # Optional: Learning Rate Scheduler (e.g., decay LR by a factor of 0.1 every 10 epochs)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # print("Using LR Scheduler: StepLR (step=10, gamma=0.1)")
    scheduler = None # Keep it simple for now

    # --- Training Loop ---
    num_epochs = config.EPOCHS
    best_model_wts = copy.deepcopy(model.state_dict()) # Keep track of best weights
    best_val_accuracy = 0.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Starting Training for {num_epochs} epochs ---")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print("Phase: Training")
            else:
                model.eval()   # Set model to evaluate mode
                print("Phase: Validation")

            running_loss = 0.0
            running_corrects = 0
            batch_count = 0

            # Iterate over data using tqdm for progress bar
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
            for inputs, labels in pbar:
                # Handle potential None returns if ECGImageDataset had errors (needs robust handling)
                if inputs is None or labels is None:
                    print("Warning: Skipping batch due to None input/label.")
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Get the index of the max log-probability
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                batch_loss = loss.item() * inputs.size(0) # Loss for the batch
                batch_corrects = torch.sum(preds == labels.data).item() # Correct predictions in batch
                running_loss += batch_loss
                running_corrects += batch_corrects
                batch_count += inputs.size(0)

                # Update tqdm progress bar description
                pbar.set_postfix({'Loss': f'{batch_loss/inputs.size(0):.4f}', 'Acc': f'{batch_corrects/inputs.size(0):.4f}'})


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            training_history[f'{phase}_loss'].append(epoch_loss)
            training_history[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy is the best we've seen so far
            if phase == 'val' and epoch_acc > best_val_accuracy:
                best_val_accuracy = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'** New best validation accuracy: {best_val_accuracy:.4f} -> Saving model weights... **')
                # --- Save the best model weights ---
                try:
                    # Ensure the save directory exists
                    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
                    torch.save(best_model_wts, config.MODEL_SAVE_PATH)
                    print(f"Best model weights saved to {config.MODEL_SAVE_PATH}")
                except Exception as e:
                    print(f"Error saving model: {e}")


        # Step the scheduler (if using one)
        if scheduler:
            scheduler.step() # Or scheduler.step(epoch_val_loss) for ReduceLROnPlateau

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")


    # --- Training Complete ---
    training_time = time.time() - start_time
    print(f'\n--- Training complete in {training_time // 60:.0f}m {training_time % 60:.0f}s ---')
    print(f'Best Validation Accuracy: {best_val_accuracy:4f}')

    # Optional: Load best model weights back into model
    # model.load_state_dict(best_model_wts)

    # Optional: Save training history (e.g., to a csv or plot it)
    history_df = pd.DataFrame(training_history)
    history_save_path = os.path.join(config.BASE_DIR, 'reports', 'training_history.csv')
    try:
        os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
        history_df.to_csv(history_save_path, index=False)
        print(f"Training history saved to {history_save_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")

    print("--- Model Training Finished ---")


# --- Script Execution ---
if __name__ == '__main__':
    train_model()