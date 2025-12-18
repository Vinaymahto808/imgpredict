# CIFAR-10 Image Classifier

A deep learning web application that classifies images into 10 categories using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

## 🚀 Live Demo

Deploy this app on Streamlit Cloud: [Your App URL Here]

## 📋 Features

- **Real-time Image Classification**: Upload any image and get instant predictions
- **10 Categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Confidence Scores**: See how confident the model is about its predictions
- **Detailed Analysis**: View probability scores for all categories
- **User-friendly Interface**: Clean and intuitive web interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: TensorFlow/Keras
- **Model**: Custom CNN with BatchNormalization and Dropout
- **Dataset**: CIFAR-10

## 📦 Installation & Local Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd img_prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## 🌐 Deploy on Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** and upload these files:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (this file)

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the details**:
   - Repository: `your-username/img_prediction`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click "Deploy"**

### Step 3: Configuration

The app will automatically:
- Install dependencies from `requirements.txt`
- Load and train the CNN model on first use
- Cache the model for subsequent uses

## 🎯 How to Use

1. **Load Model**: Click the "Load & Train Model" button (first time only)
2. **Upload Image**: Choose an image file (JPG, PNG, BMP)
3. **Get Prediction**: Click "Classify Image" to see results
4. **View Details**: Check "Show All Probabilities" for detailed analysis

## 🏗️ Model Architecture

```
Input (32x32x3)
├── Conv2D (32) + BatchNorm + ReLU
├── Conv2D (32) + BatchNorm + ReLU
├── MaxPooling2D + Dropout(0.2)
├── Conv2D (64) + BatchNorm + ReLU
├── Conv2D (64) + BatchNorm + ReLU
├── MaxPooling2D + Dropout(0.3)
├── Conv2D (128) + BatchNorm + ReLU
├── Conv2D (128) + BatchNorm + ReLU
├── MaxPooling2D + Dropout(0.4)
├── Conv2D (256) + BatchNorm + ReLU
├── Conv2D (256) + BatchNorm + ReLU
├── MaxPooling2D + Dropout(0.5)
├── Flatten
└── Dense (10) + Softmax
```

## 📊 Performance

- **Training**: Uses data augmentation and regularization
- **Validation**: Split from training data (10%)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Callbacks**: Early stopping and learning rate reduction

## 🔧 Customization

### Modify Model Architecture
Edit the `create_model()` function in `app.py` to change layers, filters, or activation functions.

### Adjust Training Parameters
Modify the `load_and_train_model()` function to change:
- Number of epochs
- Batch size
- Learning rate
- Validation split

### Change UI Elements
Update the Streamlit components in `app.py` to customize the interface.

## 📝 File Structure

```
img_prediction/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── CIFAR10_project_YT.ipynb  # Original Jupyter notebook
```

## 🚨 Important Notes

- **First Load**: The model trains on first use, which takes 2-3 minutes
- **Memory Usage**: The app uses caching to optimize performance
- **Image Size**: All images are automatically resized to 32x32 pixels
- **Supported Formats**: JPG, JPEG, PNG, BMP

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Support

If you encounter any issues or have questions:
1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include error messages and screenshots if applicable

---

**Built with ❤️ using Streamlit and TensorFlow**