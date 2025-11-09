# Skin Cancer Detection System 🔬

A comprehensive multi-model skin cancer detection system using machine learning and deep learning to classify dermatological lesions from the HAM10000 dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://skin-cancer-detection---ham10000git-9pkpmijrkce7p5jzt9ty8t.streamlit.app/)

## 📋 Overview

This project implements three different classification approaches for skin cancer detection:
- **Binary SVM Classifier**: Distinguishes between benign and malignant lesions
- **Multiclass SVM Classifier**: Identifies 7 different types of skin lesions
- **CNN Deep Learning Model**: Advanced neural network for 7-class classification with confidence scores

## 🎯 Features

- **Multi-Model Architecture**: Choose between SVM and CNN models based on your needs
- **Real-time Predictions**: Upload an image and get instant AI-powered diagnoses
- **Interactive Web Interface**: User-friendly Streamlit application
- **Confidence Scoring**: CNN model provides probability distributions across all classes
- **Medical Disclaimers**: Clear warnings about consulting healthcare professionals

## 🏥 Lesion Types Detected

The system can identify the following skin lesion types:

| Code | Full Name | Type |
|------|-----------|------|
| **nv** | Melanocytic Nevi (Nevus) | Benign |
| **mel** | Melanoma | Malignant |
| **bkl** | Benign Keratosis-like | Benign |
| **bcc** | Basal Cell Carcinoma | Malignant |
| **akiec** | Actinic Keratoses | Malignant |
| **vasc** | Vascular Lesions | Benign |
| **df** | Dermatofibroma | Benign |

## 📊 Dataset

The models are trained on the **HAM10000** (Human Against Machine with 10000 training images) dataset, which consists of:
- 10,015 dermatoscopic images
- 7 different categories of pigmented lesions
- Diverse patient demographics and lesion localizations

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn (SVM models)
- **Deep Learning**: TensorFlow/Keras (CNN)
- **Image Processing**: scikit-image, OpenCV, PIL
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure

```
skin-cancer-detection/
│
├── main_deploy.py          # Streamlit deployment application
├── main_trainning.py        # Model training pipeline
├── preprocessing.py         # Data preprocessing utilities
│
├── models/                  # Trained model files
│   ├── svm_binary_model.pkl
│   ├── svm_multiclass_model.pkl
│   ├── cnn_multi_model.h5
│   └── scaler.save
│
└── ham10000-dataset/        # Dataset (not included in repo)
    ├── HAM10000_images_part_1/
    ├── HAM10000_images_part_2/
    └── HAM10000_metadata.csv
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/HoangLongCanCode/Skin-Cancer-Detection---HAM10000.git
cd skin-cancer-detection
```

2. **Install dependencies**
```bash
pip install streamlit joblib numpy pillow tensorflow scikit-image scikit-learn pandas matplotlib seaborn opencv-python imbalanced-learn gdown
```

3. **Download the HAM10000 dataset**
- Visit [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Place the dataset in `ham10000-dataset/` directory

4. **Run the training pipeline** (optional - pre-trained models are provided)
```bash
python main_trainning.py
```

5. **Launch the web application**
```bash
streamlit run main_deploy.py
```

## 💻 Usage

### Web Application

1. Visit the [deployed application](https://skin-cancer-detection---ham10000git-9pkpmijrkce7p5jzt9ty8t.streamlit.app/)
2. Select a detection model from the sidebar:
   - Binary SVM (fastest, benign/malignant only)
   - Multiclass SVM (7 lesion types)
   - CNN Deep Learning (most accurate, with confidence scores)
3. Upload a clear image of the skin lesion (JPG, JPEG, or PNG)
4. View the AI-powered prediction results

### Local Development

```python
import streamlit as st
# Run the app
streamlit run main_deploy.py
```

## 🧠 Model Performance

### Binary SVM Classifier
- **Task**: Benign vs Malignant classification
- **Features**: Grayscale HOG features (28×28 pixels)
- **Preprocessing**: StandardScaler normalization
- **Class Balancing**: RandomOverSampler

### Multiclass SVM Classifier
- **Task**: 7-class lesion type classification
- **Kernel**: RBF (Radial Basis Function)
- **Features**: Grayscale features with dimensionality reduction
- **Class Balancing**: RandomOverSampler

### CNN Deep Learning Model
- **Architecture**: Custom convolutional neural network
- **Input**: RGB images (28×28×3)
- **Layers**: 
  - Conv2D → Conv2D → MaxPooling2D
  - Conv2D → Conv2D → MaxPooling2D
  - Flatten → Dense(64) → Dense(32) → Dense(7, softmax)
- **Training**: 20 epochs with ModelCheckpoint callback
- **Output**: Probability distribution across all 7 classes

## 📈 Training Details

The models were trained using:
- **Train/Test Split**: 80/20
- **Validation Split**: 20% of training data
- **Batch Size**: 128 (CNN)
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: ModelCheckpoint for best model saving

## ⚠️ Important Disclaimers

> **Medical Disclaimer**: This is an AI research tool and should **NOT** replace professional medical diagnosis. Always consult a qualified dermatologist for proper evaluation, diagnosis, and treatment recommendations.

- This system is for educational and research purposes only
- AI predictions should not be used as the sole basis for medical decisions
- Early detection and professional medical evaluation save lives
- If you notice any changes in your skin, seek immediate medical attention

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161.
- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/) and [TensorFlow](https://www.tensorflow.org/)

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the repository maintainer.

---

<div align="center">
<b>🔬 Skin Cancer Detection Research Project</b><br>
Multi-Model Classification System<br>
Built with Streamlit • Powered by Machine Learning & Deep Learning
</div>
