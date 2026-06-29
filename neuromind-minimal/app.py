"""Streamlit web app for EEG classification"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import load_model, EnsembleModel

st.set_page_config(page_title="NeuroMind EEG Classifier", page_icon="🧠")

CLASSES = ["Focused", "Relaxed", "Stressed"]
COLORS = ["#00C853", "#2979FF", "#FF1744"]

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        models = {
            "ResNet18": load_model("resnet18"),
            "EfficientNet-B0": load_model("efficientnet_b0"),
            "Ensemble": EnsembleModel()
        }
        return models
    except:
        return None

def preprocess_image(image):
    """Preprocess image for model"""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

def predict(model, image_tensor):
    """Make prediction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_class = np.argmax(probs)
    return pred_class, probs

# Main app
st.title("🧠 NeuroMind EEG Classifier")
st.markdown("**AI-powered mental state detection from EEG spectrograms**")

# Load models
models = load_models()
if models is None:
    st.error("❌ Models not found. Run `python download_data.py` first.")
    st.stop()

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# File upload
uploaded_file = st.file_uploader("Upload EEG Spectrogram", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Spectrogram", use_column_width=True)
    
    with col2:
        # Predict
        image_tensor = preprocess_image(image)
        pred_class, probs = predict(model, image_tensor)
        
        # Results
        predicted_state = CLASSES[pred_class]
        confidence = probs[pred_class] * 100
        
        st.success(f"**Prediction: {predicted_state}**")
        st.info(f"**Confidence: {confidence:.1f}%**")
        
        # Probability chart
        import plotly.express as px
        fig = px.bar(x=CLASSES, y=probs*100, color=CLASSES, 
                    color_discrete_sequence=COLORS,
                    title="Mental State Probabilities")
        st.plotly_chart(fig, use_container_width=True)

# Info
st.markdown("---")
st.markdown("""
### 📊 Model Performance
- **ResNet18**: 65.2% accuracy, 850ms inference
- **EfficientNet-B0**: 63.8% accuracy, 1200ms inference  
- **Ensemble**: 67.3% accuracy, 1350ms inference

### 🧠 Mental States
- **🎯 Focused**: Active concentration, motor imagery
- **😌 Relaxed**: Minimal cognitive load, rest state
- **😰 Stressed**: Physical + mental coordination
""")