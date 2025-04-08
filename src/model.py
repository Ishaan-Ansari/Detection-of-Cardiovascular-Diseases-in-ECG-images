# src/model.py

import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ------------------------------------

# Import configuration
from src import config

def create_model(num_classes=None, freeze_feature_extractor=True):
    """
    Creates a CNN model based on a pre-trained ResNet50 architecture.

    Args:
        num_classes (int, optional): Number of output classes. If None, uses
                                     config.NUM_CLASSES. Defaults to None.
        freeze_feature_extractor (bool): If True, freezes the weights of the
                                         pre-trained convolutional layers.
                                         Defaults to True.

    Returns:
        torch.nn.Module: The initialized PyTorch model.
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
        if num_classes <= 0:
             raise ValueError("Number of classes must be positive. Check config.NUM_CLASSES.")

    print(f"Creating model with {num_classes} output classes.")
    print(f"Using pre-trained ResNet50.")
    print(f"Freezing feature extractor layers: {freeze_feature_extractor}")

    # Load pre-trained ResNet50 model using the recommended weights API
    # Using IMAGENET1K_V2 weights which are often slightly better than V1
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze convolutional layers if requested
    if freeze_feature_extractor:
        for param in model.parameters():
            param.requires_grad = False
        print("Frozen base model parameters.")

    # Modify the final fully connected layer (classifier head)
    # ResNet's final layer is typically named 'fc'
    num_ftrs = model.fc.in_features
    print(f"Original ResNet50 classifier input features: {num_ftrs}")

    # Replace the existing fc layer with a new one matching our number of classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced final layer with new nn.Linear({num_ftrs}, {num_classes}).")

    # Note: If layers were frozen, only the parameters of the new 'fc' layer will have requires_grad=True
    # If freeze_feature_extractor was False, all parameters (including the new fc layer) will have requires_grad=True

    return model

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Running model.py directly for testing...")

    # Ensure config is loaded correctly
    if config.NUM_CLASSES is None or config.NUM_CLASSES <= 0:
         print("Error: config.NUM_CLASSES not set or invalid.")
         # Set a default for testing if needed, but configuration should be correct
         # config.NUM_CLASSES = 4 # Example default for testing only
         exit() # Exit if config is not properly set

    # Create the model
    test_model = create_model(num_classes=config.NUM_CLASSES, freeze_feature_extractor=True)

    # Print model summary (optional, can be very long for ResNet)
    # print("\nModel Architecture:")
    # print(test_model)

    # Check parameter trainability (if frozen)
    print("\nChecking parameter trainability (requires_grad):")
    total_params = 0
    trainable_params = 0
    for name, param in test_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(f"  - Trainable: {name}") # Uncomment to list trainable layers
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


    # Test forward pass with a dummy input batch
    print("\nTesting forward pass...")
    try:
        # Create a dummy input tensor matching expected shape [Batch, Channels, Height, Width]
        dummy_input = torch.randn(config.BATCH_SIZE,
                                  config.IMG_CHANNELS,
                                  config.IMG_HEIGHT,
                                  config.IMG_WIDTH)

        print(f"Dummy input shape: {dummy_input.shape}")

        # Perform a forward pass
        output = test_model(dummy_input)

        print(f"Output shape: {output.shape}") # Expected: [BATCH_SIZE, NUM_CLASSES]
        print("Forward pass successful.")

    except Exception as e:
        print(f"Error during forward pass test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Model Creation Test Completed ---")