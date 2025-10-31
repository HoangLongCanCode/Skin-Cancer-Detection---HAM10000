import streamlit as st
import joblib
import numpy as np
import os
import gdown
from PIL import Image
from tensorflow import keras
from skimage.color import rgb2gray
from skimage.transform import resize

# Class mappings
BINARY_CLASSES = {
    0: "Benign",
    1: "Malignant"
}

MULTICLASS_CLASSES = {
    4: ('nv', 'Melanocytic Nevi (Nevus)'), 
    6: ('mel', 'Melanoma'), 
    2: ('bkl', 'Benign Keratosis-like'), 
    1: ('bcc', 'Basal Cell Carcinoma'), 
    5: ('vasc', 'Vascular Lesions'), 
    0: ('akiec', 'Actinic Keratoses'),  
    3: ('df', 'Dermatofibroma')
}

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("🔬 Skin Cancer Detection System")
st.markdown("""
Upload a skin lesion image and select a detection model to get AI-powered predictions.
Choose from binary classification (Benign/Malignant), multiclass SVM, or deep learning CNN models.
""")

# Sidebar for model selection
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Choose Detection Model:",
    ["Binary SVM (Benign/Malignant)", "Multiclass SVM (7 Types)", "CNN Deep Learning (7 Types)"],
    help="Select which model to use for prediction"
)

# Load models and scaler

@st.cache_resource
def load_models():
    try:
        # Create a folder to store models
        os.makedirs("models", exist_ok=True)

        # Google Drive model links (convert to uc?id= format)
        drive_links = {
            "svm_binary_model.pkl": "https://drive.google.com/uc?id=1C9QhOGIj0aXPTRNjl2CQibP1DQHLp6TW",
            "svm_multiclass_model.pkl": "https://drive.google.com/uc?id=1fkUX_58CiE8OuK-rIvuqyng6FifMQGPa",
            "cnn_multi_model.h5": "https://drive.google.com/uc?id=1LuR-DIeXkXibNTPlr6GwyIS79VV6G_XF",
            "scaler.save": "https://drive.google.com/uc?id=1Ut2iC-tVTBf3N0Ydu-tYzt8z26YQaxnn"
        }

        # Download if not already exists
        for filename, url in drive_links.items():
            local_path = os.path.join("models", filename)
            if not os.path.exists(local_path):
                st.info(f"📥 Downloading {filename} ...")
                gdown.download(url, local_path, quiet=False)
            else:
                print(f"✅ {filename} already exists")

        # Load models from the downloaded files
        binary_model = joblib.load("models/svm_binary_model.pkl")
        multiclass_svm_model = joblib.load("models/svm_multiclass_model.pkl")
        cnn_model = keras.models.load_model("models/cnn_multi_model.h5")
        scaler = joblib.load("models/scaler.save")

        st.success("✅ All models loaded successfully!")
        return binary_model, multiclass_svm_model, cnn_model, scaler

    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please ensure all model files are uploaded or accessible.")
        return None, None, None, None

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

# Image preprocessing functions
def preprocess_for_svm(image, target_size=(28, 28)):
    """Preprocess image for SVM models - grayscale and flattened"""
    from skimage.color import rgb2gray
    from skimage.transform import resize
    
    img_array = np.array(image)
    
    # Convert to grayscale if RGB
    if img_array.ndim == 3:
        img_array = rgb2gray(img_array)
    
    # Resize image
    img_resized = resize(img_array, target_size, anti_aliasing=True)
    
    # Flatten into 1D array
    feature_vector = img_resized.flatten().reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)  # Apply scaling
    
    return feature_vector

def preprocess_for_cnn(image, target_size=(28, 28)):
    """Preprocess image for CNN model - RGB and normalized"""
    from skimage.transform import resize
    
    img_array = np.array(image)
    
    # Handle grayscale or RGBA images
    if img_array.ndim == 2:
        # Grayscale → RGB
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 4:
        # RGBA → RGB
        img_array = img_array[..., :3]
    
    # Resize image
    img_resized = resize(img_array, target_size, anti_aliasing=True)
    
    # Normalize pixel values to [0, 1]
    img_resized = img_resized / 255.0 if img_resized.max() > 1 else img_resized
    
    img_ready = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    
    return img_ready

# Load models
binary_model, multiclass_svm_model, cnn_model, scaler = load_models()

if binary_model is None or multiclass_svm_model is None or cnn_model is None:
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the skin lesion"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Show image info
        st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.header("🔍 Prediction Results")
    
    if uploaded_file is not None:
        with st.spinner("🔄 Analyzing image..."):
            try:
                # Choose model and predict based on selection
                if model_choice == "Binary SVM (Benign/Malignant)":
                    # Preprocess for SVM
                    processed_image = preprocess_for_svm(image)
                    
                    # Predict
                    prediction = int(binary_model.predict(processed_image)[0])
                    
                    # Display result
                    st.subheader("🔵 Binary Classification Result")
                    result = BINARY_CLASSES[prediction]
                    
                    if prediction == 1:  # Malignant
                        st.error(f"### ⚠️ {result}")
                        st.warning("The lesion appears to be **malignant**. Please consult a dermatologist immediately.")
                    else:  # Benign
                        st.success(f"### ✅ {result}")
                        st.info("The lesion appears to be **benign**. However, regular monitoring is still recommended.")
                
                elif model_choice == "Multiclass SVM (7 Types)":
                    # Preprocess for SVM
                    processed_image = preprocess_for_svm(image)
                    
                    # Predict
                    prediction = int(multiclass_svm_model.predict(processed_image)[0])
                    
                    # Display result
                    st.subheader("🔬 Multiclass SVM Result")
                    abbrev, full_name = MULTICLASS_CLASSES[prediction]
                    
                    st.info(f"### 🎯 {full_name}")
                    st.write(f"**Abbreviation:** {abbrev}")
                    st.write(f"**Predicted Class:** {prediction}")
                    
                    # Add severity indicator
                    malignant_types = {6, 0, 1}  # mel, akiec, bcc
                    if prediction in malignant_types:
                        st.error("⚠️ This type requires medical attention")
                    else:
                        st.success("✓ This type is typically benign")
                
                else:  # CNN Deep Learning
                    # Preprocess for CNN
                    processed_image = preprocess_for_cnn(image)
                    
                    # Predict
                    predictions = cnn_model.predict(processed_image, verbose=0)
                    prediction_probs = predictions[0]
                    predicted_class = int(np.argmax(prediction_probs))
                    confidence = float(prediction_probs[predicted_class]) * 100
                    
                    # Display result
                    st.subheader("🧠 CNN Deep Learning Result")
                    abbrev, full_name = MULTICLASS_CLASSES[predicted_class]
                    
                    st.info(f"### 🎯 {full_name}")
                    st.write(f"**Abbreviation:** {abbrev}")
                    st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Show probability distribution
                    st.write("**Probability Distribution:**")
                    
                    # Sort by probability
                    sorted_indices = np.argsort(prediction_probs)[::-1]
                    
                    for idx in sorted_indices[:3]:  # Show top 3
                        class_abbrev, class_name = MULTICLASS_CLASSES[idx]
                        prob = prediction_probs[idx] * 100
                        st.progress(prob / 100)
                        st.write(f"{class_name}: {prob:.2f}%")
                    
                    # Add severity indicator
                    malignant_types = {6, 0, 1}  # mel, akiec, bcc
                    if predicted_class in malignant_types:
                        st.error("⚠️ This type requires medical attention")
                    else:
                        st.success("✓ This type is typically benign")
                
                # Medical disclaimer
                st.divider()
                st.warning("""
                    ⚠️ **Important Medical Disclaimer:**
                    
                    This is an AI research tool and should **NOT** replace professional medical diagnosis.
                    
                    **Always consult a qualified dermatologist for:**
                    - Proper evaluation and diagnosis
                    - Treatment recommendations
                    - Regular skin cancer screenings
                    
                    Early detection saves lives. If you notice any changes in your skin, seek medical attention.
                """)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("Please try uploading a different image or contact support.")
    
    else:
        st.info("👆 Please upload an image to get started")
        
        # Show example of what to upload
        st.markdown("""
        ### 📋 Image Guidelines:
        - Clear, well-lit photo of the skin lesion
        - Close-up view of the affected area
        - Avoid blurry or dark images
        - Supported formats: JPG, JPEG, PNG
        """)

# Sidebar information
st.sidebar.divider()
st.sidebar.header("ℹ️ Model Information")

if model_choice == "Binary SVM (Benign/Malignant)":
    st.sidebar.info("""
    **Binary SVM Model**
    - Classifies as Benign or Malignant
    - Fast prediction
    - Good for initial screening
    """)
elif model_choice == "Multiclass SVM (7 Types)":
    st.sidebar.info("""
    **Multiclass SVM Model**
    - Identifies 7 different lesion types
    - Uses classical machine learning
    - Trained on HOG features
    """)
else:
    st.sidebar.info("""
    **CNN Deep Learning Model**
    - Advanced neural network
    - Identifies 7 different lesion types
    - Provides confidence scores
    - Most accurate predictions
    """)

st.sidebar.divider()
st.sidebar.markdown("""
### 🏥 Lesion Types:
- **nv**: Melanocytic Nevi (moles)
- **mel**: Melanoma (cancer)
- **bkl**: Benign Keratosis
- **bcc**: Basal Cell Carcinoma
- **akiec**: Actinic Keratoses
- **vasc**: Vascular Lesions
- **df**: Dermatofibroma
""")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p><small>🔬 Skin Cancer Detection Research Project | Multi-Model Classification System</small></p>
    <p><small>Built with Streamlit • Powered by Machine Learning & Deep Learning</small></p>
</div>
""", unsafe_allow_html=True)