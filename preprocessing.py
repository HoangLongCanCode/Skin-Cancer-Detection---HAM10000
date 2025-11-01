import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm

import matplotlib.image as mpimg
from skimage.feature import hog
from skimage.transform import resize
from skimage.color import rgb2gray

#Mapping Diagnose Dictionary
dx_dict = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-Like',
    'df': 'Dermatofibroma',
    'nv': 'Nevus',
    'mel': 'Melanoma',
    'vasc': 'Vascular Lesions'
}

dx_binary = {
    'akiec': '1',
    'bcc': '1',
    'bkl': '0',
    'df': '0',
    'nv': '0',
    'mel': '1',
    'vasc': '1'
}

def exploratory_data_analysis(tabular_data):
    #Count values
    print("Value counts for categorical features:\n")
    print(tabular_data['dx'].value_counts(), "\n")
    print(tabular_data['dx_type'].value_counts(), "\n")
    print(tabular_data['sex'].value_counts(), "\n")
    print(tabular_data['localization'].value_counts(), "\n")

    #Distribution plots
    plt.figure(figsize=(8,4))
    sns.countplot(x='dx', data=tabular_data, order=tabular_data['dx'].value_counts().index, palette='viridis')
    plt.title('Distribution of Diagnosis (dx)')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(6,4))
    sns.countplot(x='dx_type', data=tabular_data, palette='mako')
    plt.title('Distribution of Diagnosis Type (dx_type)')
    plt.xlabel('Diagnosis Type')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(5,4))
    sns.countplot(x='sex', data=tabular_data, palette='coolwarm')
    plt.title('Sex Distribution')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8,4))
    sns.histplot(tabular_data['age'], bins=30, kde=True, color='teal')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10,5))
    sns.countplot(x='localization', data=tabular_data, order=tabular_data['localization'].value_counts().index, palette='cubehelix')
    plt.title('Lesion Localization Distribution')
    plt.xlabel('Body Location')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

def preprocess_values(tabular_data):
    print("=== Checking missing values ===")
    print(tabular_data.isnull().sum(), "\n")

    # Handle missing values in 'age'
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    tabular_data["age"] = imputer.fit_transform(tabular_data[["age"]])
    print(tabular_data.isnull().sum(), "\n")
    print("✅ Missing values handled.\n")

    # Map categorical values
    print("=== Mapping diagnosis values ===")
    tabular_data["diagnosis"] = tabular_data["dx"].map(dx_dict)
    tabular_data["malignant"] = tabular_data["dx"].map(dx_binary)
    print("✅ Diagnosis and malignancy columns added.\n")

    # Encode categorical features
    cat_cols = ["dx_type", "sex", "localization"]
    encoders = {}
    print("=== Encoding categorical columns ===")
    for col in cat_cols:
        le = LabelEncoder()
        tabular_data[col] = le.fit_transform(tabular_data[col])
        encoders[col] = le
        print(f"Encoded '{col}' with classes: {list(le.classes_)}")
    print("✅ All categorical columns encoded.\n")

    # Standardize numerical feature
    print("=== Standardizing numerical features ===")
    scaler = StandardScaler()
    tabular_data["age"] = scaler.fit_transform(tabular_data[["age"]])
    print("✅ Standardization complete.\n")

    print("=== Final preview ===")
    print(tabular_data.head(), "\n")

    return tabular_data, encoders

import os

def get_image_paths(folder_list, extensions={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}):
    """
    Collect all image file paths from multiple folders.

    Parameters:
    -----------
    folder_list : list
        List of folder paths
    extensions : set
        Allowed image file extensions (default common formats)

    Returns:
    --------
    img_paths : list
        List of full image file paths
    """

    img_paths = []

    for folder in folder_list:
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                img_paths.append(os.path.join(folder, filename))

    print(f"✅ Found {len(img_paths)} images in {len(folder_list)} folders.")
    return img_paths


def extract_grayscale_features(image_paths, target_size=(28, 28)):
    features = []
    print(f"Extracting features from {len(image_paths)} images...")

    for i, path in enumerate(image_paths):
        try:
            img = mpimg.imread(path)

            # Convert to grayscale if RGB
            if img.ndim == 3:
                img = rgb2gray(img)

            # Resize image
            img_resized = resize(img, target_size, anti_aliasing=True)

            # Flatten into 1D array
            feature_vector = img_resized.flatten()
            features.append(feature_vector)

            # Print progress every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")

        except Exception as e:
            print(f"⚠️ Error processing image {path}: {e}")
            continue

    features = np.array(features)
    print(f"\n✅ Feature extraction complete! Shape: {features.shape}")
    return features

def extract_rgb_features(image_paths, target_size=(28, 28)):
    """
    Extract RGB features for CNN training (keeps 3 color channels).
    Returns a 4D numpy array of shape (N, H, W, 3)
    """
    features = []
    print(f"Extracting RGB features from {len(image_paths)} images...")

    for i, path in enumerate(image_paths):
        try:
            # Read image
            img = mpimg.imread(path)

            # Handle grayscale or RGBA images
            if img.ndim == 2:
                # Grayscale → RGB
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] == 4:
                # RGBA → RGB
                img = img[..., :3]

            # Resize image
            img_resized = resize(img, target_size, anti_aliasing=True)

            # Normalize pixel values to [0, 1]
            img_resized = img_resized / 255.0 if img_resized.max() > 1 else img_resized

            feature_vector = img_resized.flatten()
            features.append(feature_vector)

            # Print progress every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")

        except Exception as e:
            print(f"⚠️ Error processing image {path}: {e}")
            continue

    features = np.array(features, dtype=np.float32)
    print(f"\n✅ RGB feature extraction complete! Shape: {features.shape}")
    return features