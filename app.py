import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import pickle
import os

# Page config
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="📷",
    layout="centered"
)

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def create_model():
    """Create and return the CNN model architecture"""
    model = Sequential()
    weight_decay = 0.0001
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.3))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.4))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', 
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.5))
    
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    return model

@st.cache_data
def load_and_train_model():
    """Load CIFAR-10 data, train model, and return model with preprocessing params"""
    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=101)
    
    # Preprocessing
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    
    mean = np.mean(X_train)
    std = np.std(X_train)
    
    X_train = (X_train - mean) / (std + 0.00000001)
    X_valid = (X_valid - mean) / (std + 0.00000001)
    
    y_train = to_categorical(y_train, 10)
    y_valid = to_categorical(y_valid, 10)
    
    # Create and compile model
    model = create_model()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model (reduced epochs for demo)
    with st.spinner("Training model... This may take a few minutes."):
        model.fit(X_train, y_train, 
                 validation_data=(X_valid, y_valid),
                 epochs=5, 
                 batch_size=64, 
                 verbose=0)
    
    return model, mean, std

def preprocess_image(image, mean, std):
    """Preprocess image for prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((32, 32))
    image_array = np.array(image, dtype=np.float32)
    normalized_image = (image_array - mean) / (std + 0.00000001)
    
    return normalized_image.reshape((1, 32, 32, 3))

# App title
st.title("🖼️ CIFAR-10 Image Classifier")
st.markdown("Upload an image to classify it into one of the **CIFAR-10 categories**")

# Display categories
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Categories:**")
    for i in range(0, 5):
        st.write(f"• {class_names[i]}")
with col2:
    st.write("")
    for i in range(5, 10):
        st.write(f"• {class_names[i]}")

st.markdown("---")

# Initialize model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    if st.button("🚀 Load & Train Model", type="primary"):
        try:
            model, mean, std = load_and_train_model()
            st.session_state.model = model
            st.session_state.mean = mean
            st.session_state.std = std
            st.session_state.model_loaded = True
            st.success("✅ Model loaded and trained successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
else:
    st.success("✅ Model is ready!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image in JPG, PNG, or BMP format"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Uploaded Image")
            st.image(image, caption=f"File: {uploaded_file.name}", use_column_width=True)
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        processed_image = preprocess_image(image, st.session_state.mean, st.session_state.std)
                        prediction = st.session_state.model.predict(processed_image, verbose=0)
                        predicted_class_idx = np.argmax(prediction)
                        predicted_class = class_names[predicted_class_idx]
                        confidence = float(np.max(prediction))
                        
                        st.success(f"**Prediction:** {predicted_class.upper()}")
                        st.info(f"**Confidence:** {confidence:.1%}")
                        st.progress(confidence)
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
        
        # Show all probabilities
        if st.button("📊 Show All Probabilities"):
            with st.spinner("Computing probabilities..."):
                try:
                    processed_image = preprocess_image(image, st.session_state.mean, st.session_state.std)
                    prediction = st.session_state.model.predict(processed_image, verbose=0)
                    
                    st.subheader("📈 Prediction Probabilities")
                    
                    prob_data = {}
                    for i, class_name in enumerate(class_names):
                        prob_data[class_name] = float(prediction[0][i])
                    
                    st.bar_chart(prob_data)
                    
                    sorted_probs = sorted(prob_data.items(), key=lambda x: x[1], reverse=True)
                    
                    st.subheader("🏆 Ranked Results")
                    for i, (class_name, prob) in enumerate(sorted_probs):
                        rank = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
                        st.write(f"{rank}: **{class_name}**: {prob:.1%}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This app uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset "
    "to classify images into 10 different categories."
)

st.sidebar.header("📋 How to Use")
st.sidebar.markdown("""
1. **Load Model**: Click 'Load & Train Model' button
2. **Upload**: Choose an image file
3. **Classify**: Click 'Classify Image' for prediction
4. **Explore**: View detailed probabilities
""")

st.sidebar.header("🔧 Model Info")
if st.session_state.get('model_loaded', False):
    st.sidebar.success("Status: ✅ Loaded")
    try:
        total_params = st.session_state.model.count_params()
        st.sidebar.info(f"Parameters: {total_params:,}")
    except:
        pass
else:
    st.sidebar.error("Status: ❌ Not Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit & TensorFlow*")