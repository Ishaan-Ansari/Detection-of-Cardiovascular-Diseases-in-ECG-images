# ECG Image Classification for Cardiovascular Diseases

## Project Overview

This project aims to detect cardiovascular diseases by classifying Electrocardiogram (ECG) images using deep learning. The primary goal is to build a model that can accurately classify an input ECG image into one of the following categories:

1.  **Normal:** ECG from a healthy individual.
2.  **Myocardial Infarction (MI):** ECG showing signs of a heart attack.
3.  **Abnormal Heartbeat:** ECG indicating arrhythmias or other heartbeat irregularities.
4.  **History of Myocardial Infarction:** ECG showing signs of a past heart attack.


## Project Structure

```text
ecg_classification/
│
├── data/
│   └── raw/                    # Place raw class‑sorted images here (e.g., data/raw/Normal/*.jpg)
│
├── models/                     # Saved trained model weights (e.g., .pth files)
├── reports/                    # Evaluation results, figures, training history
│   ├── figures/                # Saved plots (e.g., confusion matrix)
│   └── training_history.csv    # CSV log of training/validation metrics
│
├── src/                        # Source code modules
│   ├── config.py               # Configuration (paths, hyperparameters)
│   ├── data_loader.py          # Data loading & splitting logic
│   ├── preprocessing.py        # Data preprocessing & augmentation
│   ├── model.py                # CNN model architecture (PyTorch)
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluate trained model on test set
│
├── venv/                       # Python virtual environment
│
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Dataset

* **Type:** ECG Images (e.g., JPG, PNG format).
* **Source:** (Please specify the source of your data if possible/applicable).
* **Structure:** The raw image data should be organized into subdirectories within `data/raw/`, where each subdirectory name corresponds exactly to a class label. For example:
    * `data/raw/Normal/`
    * `data/raw/Myocardial Infarction/`
    * `data/raw/Abnormal Heartbeat/`
    * `data/raw/History of MI/`
    The `data_loader.py` script automatically detects classes based on these folder names.

ECG dataset: https://data.mendeley.com/datasets/gwbz3fsgp8/2

## Requirements

* Python 3.8+
* PyTorch (including torchvision, torchaudio)
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Tqdm
* Pillow (PIL Fork)
* Streamlit

All necessary Python packages are listed in `requirements.txt`.

## Setup Instructions

1.  **Clone Repository (if applicable):**
    ```bash
    # git clone https://github.com/Ishaan-Ansari/Detection-of-Cardiovascular-Diseases-in-ECG-images.git
    # cd ecg_classification
    ```
    (If you already have the files, just navigate to the project root directory `ecg_classification/`)

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    ```
    * *On macOS/Linux:* `source venv/bin/activate`
    * *On Windows (CMD):* `.\venv\Scripts\activate`
    * *On Windows (PowerShell):* `.\venv\Scripts\Activate.ps1`

3.  **Install Dependencies:**
    * **Important:** Check the official PyTorch website ([pytorch.org](https://pytorch.org/get-started/locally/)) for the correct installation command based on your OS and CUDA version (if using GPU). Install PyTorch first using their command.
    * Then, install the remaining packages:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Prepare Data:**
    * Place your organized ECG image dataset into the `data/raw/` directory as described in the [Dataset](#dataset) section.

## How to Run

Ensure your virtual environment is activated before running any commands. All commands should be run from the project's root directory (`ecg_classification/`).

1.  **Train the Model:**
    ```bash
    python src/train.py
    ```
    * This script will:
        * Load and split the data from `data/raw/`.
        * Build the ResNet50-based model.
        * Train the model using parameters from `src/config.py`.
        * Print training and validation progress per epoch.
        * Save the best performing model weights (based on validation accuracy) to the `models/` directory (e.g., `models/ecg_cnn_standard_v1.keras`).
        * Save the training history metrics to `reports/training_history.csv`.

2.  **Evaluate the Model:**
    * Make sure training has completed and a model file exists in `models/`.
    ```bash
    python src/evaluate.py
    ```
    * This script will:
        * Load the best saved model from `models/`.
        * Evaluate the model on the *test set*.
        * Print the final test accuracy, classification report (precision, recall, F1-score per class), and confusion matrix to the console.
        * Save a plot of the confusion matrix to `reports/figures/confusion_matrix.png`.

3.  **Run the Streamlit Web Application:**
    * Make sure training has completed and a model file exists in `models/`.
    ```bash
    streamlit run app.py
    ```
    * This command will start a local web server and automatically open the application in your default web browser.
    * You can upload an ECG image through the interface, and the app will display the model's predicted class and confidence score.

## Configuration

Key parameters like image dimensions, batch size, epochs, learning rate, file paths, and data augmentation settings can be modified in the `src/config.py` file.


