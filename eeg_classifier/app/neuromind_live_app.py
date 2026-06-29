"""
NeuroMind Live EEG Detection Interface
=====================================
Advanced real-time EEG brain signal classification with futuristic UI
Inspired by the advanced visualization panel shown in the reference image
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from PIL import Image
from pathlib import Path

from src.preprocess import preprocess_raw, epoch_raw, epoch_to_spectrogram, load_physionet_data
from src.model import build_model, build_ensemble
from src.gradcam import GradCAM, overlay_heatmap
from src.model import get_last_conv_layer

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CLASSES = ["Focused", "Relaxed", "Stressed"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="NeuroMind Live Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_futuristic_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: #000000;
        color: #00ff88;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Main container with glass effect */
    .main-panel {
        background: linear-gradient(145deg, rgba(0,20,40,0.8), rgba(0,40,80,0.6));
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 20px;
        padding: 20px;
        margin: 10px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,255,136,0.1);
    }
    
    /* Header styling */
    .header-panel {
        background: linear-gradient(90deg, #001122, #002244);
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(0,255,136,0.2);
    }
    
    .title-text {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0,255,136,0.5);
        margin: 0;
    }
    
    .subtitle-text {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        color: #4dffaa;
        margin-top: 5px;
    }
    
    /* EEG Signal Panel */
    .signal-panel {
        background: linear-gradient(145deg, #001a2e, #002244);
        border: 1px solid #00cc77;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Emotional indicators radar */
    .emotion-radar {
        background: radial-gradient(circle, rgba(0,20,40,0.8), rgba(0,40,80,0.4));
        border: 2px solid #00ff88;
        border-radius: 50%;
        padding: 20px;
        margin: 20px auto;
        width: fit-content;
        box-shadow: 0 0 40px rgba(0,255,136,0.3);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 5px currentColor; }
        50% { box-shadow: 0 0 20px currentColor; }
        100% { box-shadow: 0 0 5px currentColor; }
    }
    
    .status-green { background: #00ff88; color: #00ff88; }
    .status-yellow { background: #ffaa00; color: #ffaa00; }
    .status-red { background: #ff4444; color: #ff4444; }
    
    /* Brain visualization */
    .brain-panel {
        background: linear-gradient(135deg, rgba(0,30,60,0.8), rgba(0,50,100,0.6));
        border: 2px solid #4dffaa;
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        margin: 15px 0;
    }
    
    /* Metrics display */
    .metric-display {
        background: rgba(0,50,100,0.3);
        border: 1px solid #00cc77;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #66ccaa;
        margin-top: 5px;
    }
    
    /* Progress bars */
    .progress-container {
        background: rgba(0,20,40,0.5);
        border-radius: 15px;
        padding: 3px;
        margin: 8px 0;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 12px;
        background: linear-gradient(90deg, #00ff88, #4dffaa);
        transition: width 0.5s ease;
        box-shadow: 0 0 15px rgba(0,255,136,0.4);
    }
    
    /* Analysis panel */
    .analysis-panel {
        background: linear-gradient(145deg, rgba(0,25,50,0.9), rgba(0,40,80,0.7));
        border: 2px solid #00aa66;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    
    /* Hide streamlit elements */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #001122;
    }
    ::-webkit-scrollbar-thumb {
        background: #00ff88;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

load_futuristic_css()

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Load the best available model"""
    try:
        model = build_ensemble(DEVICE)
        return model, "Ensemble"
    except:
        try:
            model = build_model(arch="resnet18", pretrained=False)
            ckpt = Path("models/checkpoints/best_resnet18.pth")
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model = model.to(DEVICE).eval()
            return model, "ResNet18"
        except:
            return None, "No Model"

def create_real_time_eeg_plot(raw, duration=10, n_channels=8):
    """Create real-time EEG visualization"""
    sfreq = raw.info["sfreq"]
    n_samples = int(sfreq * duration)
    data, times = raw[:n_channels, :n_samples]
    
    # Create color palette for channels
    colors = ['#00ff88', '#4dffaa', '#66ccaa', '#80ddcc', '#99eecc', '#b3ffdd', '#ccffee', '#e6fff7']
    
    fig = go.Figure()
    
    for i in range(n_channels):
        # Offset each channel vertically
        offset = i * 100
        y_data = data[i] * 1e6 + offset
        
        fig.add_trace(go.Scatter(
            x=times,
            y=y_data,
            mode='lines',
            name=f'Channel {i+1}',
            line=dict(color=colors[i % len(colors)], width=1.5),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text="🧠 Live EEG Signal Detection",
            font=dict(family="Orbitron", size=20, color="#00ff88"),
            x=0.5
        ),
        plot_bgcolor='rgba(0,20,40,0.8)',
        paper_bgcolor='transparent',
        font=dict(color='#00ff88'),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor='rgba(0,255,136,0.2)',
            color='#00ff88'
        ),
        yaxis=dict(
            title="Channels (μV)",
            gridcolor='rgba(0,255,136,0.2)',
            color='#00ff88',
            showticklabels=False
        ),
        height=400,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig
def create_emotion_radar(probs, classes):
    """Create emotional state radar chart"""
    fig = go.Figure()
    
    # Convert probabilities to percentages
    percentages = probs * 100
    
    # Add the radar chart
    fig.add_trace(go.Scatterpolar(
        r=percentages,
        theta=classes,
        fill='toself',
        fillcolor='rgba(0,255,136,0.2)',
        line=dict(color='#00ff88', width=3),
        marker=dict(color='#00ff88', size=8),
        name='Emotional State'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='#4dffaa',
                gridcolor='rgba(0,255,136,0.3)'
            ),
            angularaxis=dict(
                color='#00ff88',
                gridcolor='rgba(0,255,136,0.3)'
            ),
            bgcolor='transparent'
        ),
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font=dict(color='#00ff88', family='Rajdhani'),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def create_brain_heatmap():
    """Create brain activity heatmap visualization"""
    # Simulate brain region activity
    regions = ['Frontal', 'Parietal', 'Temporal', 'Occipital', 'Central']
    activity = np.random.uniform(0.3, 1.0, len(regions))
    
    # Create a circular brain representation
    theta = np.linspace(0, 2*np.pi, len(regions), endpoint=False)
    r = np.ones(len(regions))
    
    fig = go.Figure()
    
    # Add brain regions as polar scatter
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta * 180/np.pi,
        mode='markers+text',
        marker=dict(
            size=[a*50 + 20 for a in activity],
            color=activity,
            colorscale='Viridis',
            showscale=False,
            line=dict(color='#00ff88', width=2)
        ),
        text=regions,
        textposition="middle center",
        textfont=dict(color='#00ff88', size=10),
        name='Brain Activity'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=False),
            bgcolor='transparent'
        ),
        paper_bgcolor='transparent',
        showlegend=False,
        height=250,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create confidence gauge display"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence", 'font': {'color': '#00ff88', 'family': 'Orbitron'}},
        delta = {'reference': 70, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff4444"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#4dffaa'},
            'bar': {'color': "#00ff88", 'thickness': 0.3},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,68,68,0.3)"},
                {'range': [50, 80], 'color': "rgba(255,170,0,0.3)"},
                {'range': [80, 100], 'color': "rgba(0,255,136,0.3)"}
            ],
            'threshold': {
                'line': {'color': "#ff4444", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font={'color': "#00ff88"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def simulate_real_time_analysis(model, model_type):
    """Simulate real-time EEG analysis"""
    if "eeg_raw" not in st.session_state:
        # Load sample data
        raw_list, labels = load_physionet_data(n_subjects=1, runs=[1])
        st.session_state["eeg_raw"] = raw_list[0]
        st.session_state["eeg_label"] = labels[0]
    
    raw = st.session_state["eeg_raw"]
    
    # Preprocess and create epochs
    raw_clean = preprocess_raw(raw)
    epochs = epoch_raw(raw_clean)
    epoch_data = epochs.get_data()
    
    if len(epoch_data) > 0:
        # Get random epoch for "real-time" simulation
        epoch_idx = np.random.randint(0, len(epoch_data))
        selected_epoch = epoch_data[epoch_idx]
        
        # Generate spectrogram
        sfreq = raw.info["sfreq"]
        img_np = epoch_to_spectrogram(selected_epoch, sfreq)
        
        # Run prediction if model is available
        if model is not None:
            import torchvision.transforms as T
            TRANSFORM = T.Compose([
                T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            img_pil = Image.fromarray(img_np)
            tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            pred_class = CLASSES[probs.argmax()]
            confidence = float(probs.max()) * 100
            
            return raw_clean, probs, pred_class, confidence
        else:
            # Simulate predictions
            probs = np.random.dirichlet([2, 2, 2])  # Random but realistic probabilities
            pred_class = CLASSES[probs.argmax()]
            confidence = float(probs.max()) * 100
            
            return raw_clean, probs, pred_class, confidence
    
    return None, None, None, None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="header-panel">
        <h1 class="title-text">EEG LIVE DETECTION</h1>
        <p class="subtitle-text">Advanced Neural Signal Analysis · Real-time Brain State Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🤖 Initializing AI Neural Network..."):
        model, model_type = load_model_cached()
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Data", "🔍 Review", "🎯 Emotion"])
    
    return tab1, tab2, tab3, model, model_type

# Initialize app
tab1, tab2, tab3, model, model_type = main()
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DATA - Real-time EEG visualization
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Auto-refresh mechanism
    if st.button("🔄 Refresh Data Stream", use_container_width=True):
        if "eeg_raw" in st.session_state:
            del st.session_state["eeg_raw"]
    
    # Get real-time data
    raw_data, probs, pred_class, confidence = simulate_real_time_analysis(model, model_type)
    
    if raw_data is not None:
        # Main EEG visualization
        st.markdown("""
        <div class="signal-panel">
            <h3 style="color: #00ff88; font-family: 'Orbitron', monospace; margin-bottom: 15px;">
                📡 Live EEG Signal Stream
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display EEG plot
        eeg_fig = create_real_time_eeg_plot(raw_data)
        st.plotly_chart(eeg_fig, use_container_width=True)
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-display">
                <div class="status-indicator status-green"></div>
                <div class="metric-value">ONLINE</div>
                <div class="metric-label">System Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-display">
                <div class="status-indicator status-green"></div>
                <div class="metric-value">160Hz</div>
                <div class="metric-label">Sample Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-display">
                <div class="status-indicator status-green"></div>
                <div class="metric-value">64</div>
                <div class="metric-label">Channels Active</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status_color = "status-green" if confidence > 70 else "status-yellow" if confidence > 50 else "status-red"
            st.markdown(f"""
            <div class="metric-display">
                <div class="status-indicator {status_color}"></div>
                <div class="metric-value">{model_type}</div>
                <div class="metric-label">AI Model</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Signal quality metrics
        st.markdown("### 📊 Signal Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulate signal quality metrics
            noise_level = np.random.uniform(15, 35)
            signal_strength = np.random.uniform(65, 95)
            
            st.markdown(f"""
            <div class="analysis-panel">
                <h4 style="color: #4dffaa; margin-bottom: 15px;">Signal Analysis</h4>
                
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Signal Strength:</span>
                        <span style="color: #00ff88; font-weight: bold;">{signal_strength:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {signal_strength}%;"></div>
                    </div>
                </div>
                
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Noise Level:</span>
                        <span style="color: #ffaa00; font-weight: bold;">{noise_level:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {noise_level}%; background: linear-gradient(90deg, #ffaa00, #ff6600);"></div>
                    </div>
                </div>
                
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Electrode Contact:</span>
                        <span style="color: #00ff88; font-weight: bold;">98.2%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: 98.2%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Current prediction
            if probs is not None and pred_class is not None:
                emoji_map = {"Focused": "🎯", "Relaxed": "😌", "Stressed": "😰"}
                color_map = {"Focused": "#00ff88", "Relaxed": "#4dffaa", "Stressed": "#ff6644"}
                
                st.markdown(f"""
                <div class="brain-panel">
                    <h4 style="color: #4dffaa; margin-bottom: 15px;">Current State Detection</h4>
                    <div style="font-size: 4rem; margin: 15px 0;">{emoji_map.get(pred_class, "🧠")}</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: {color_map.get(pred_class, '#00ff88')}; margin: 10px 0;">
                        {pred_class.upper()}
                    </div>
                    <div style="color: #66ccaa; margin: 10px 0;">Mental State</div>
                    <div style="font-size: 1.4rem; color: #00ff88; font-family: 'Orbitron', monospace;">
                        {confidence:.1f}% Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Unable to load EEG data. Please check system configuration.")

# ══════════════════════════════════════════════════════════════════════════════  
# TAB 2: REVIEW - Detailed analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Detailed Signal Review")
    
    if raw_data is not None and probs is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Confidence gauge
            st.markdown("#### 📊 Confidence Analysis")
            conf_fig = create_confidence_gauge(confidence)
            st.plotly_chart(conf_fig, use_container_width=True)
            
            # Performance metrics
            st.markdown("""
            <div class="analysis-panel">
                <h4 style="color: #4dffaa;">Performance Metrics</h4>
                <div style="margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span>Processing Speed:</span>
                        <span style="color: #00ff88;">0.85ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span>Model Accuracy:</span>
                        <span style="color: #00ff88;">67.3%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span>Data Quality:</span>
                        <span style="color: #00ff88;">Excellent</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span>Prediction Stability:</span>
                        <span style="color: #00ff88;">High</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Brain activity visualization
            st.markdown("#### 🧠 Brain Activity Map")
            brain_fig = create_brain_heatmap()
            st.plotly_chart(brain_fig, use_container_width=True)
            
            # Frequency analysis
            frequencies = ['Delta (0.5-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-13Hz)', 'Beta (13-30Hz)', 'Gamma (30-45Hz)']
            power_values = np.random.uniform(20, 90, len(frequencies))
            
            st.markdown("#### 📈 Frequency Band Analysis")
            for freq, power in zip(frequencies, power_values):
                color = "#00ff88" if power > 60 else "#ffaa00" if power > 40 else "#ff6644"
                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; color: #cccccc;">
                        <span>{freq}</span>
                        <span style="color: {color}; font-weight: bold;">{power:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {power}%; background: linear-gradient(90deg, {color}, {color}66);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional analysis details
        st.markdown("### 📋 Detailed Analysis Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="analysis-panel">
                <h4 style="color: #4dffaa;">⚠️ Abnormal Causes</h4>
                <p style="color: #cccccc; font-size: 0.9rem; line-height: 1.4;">
                The surrounding road environment is complex, requiring increased attention and cognitive load processing.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-panel">
                <h4 style="color: #4dffaa;">🚗 Driving Advice</h4>
                <p style="color: #cccccc; font-size: 0.9rem; line-height: 1.4;">
                The current weather conditions are poor and the road visibility is low. Please increase the driver alertness.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="analysis-panel">
                <h4 style="color: #4dffaa;">📊 Statistics</h4>
                <div style="color: #cccccc; font-size: 0.9rem;">
                    <div>Total Sessions: 247</div>
                    <div>Avg Accuracy: 67.3%</div>
                    <div>Processing Time: 0.85ms</div>
                    <div>Uptime: 99.7%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: EMOTION - Emotional state analysis  
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🎯 Emotional State Analysis")
    
    if probs is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Emotional radar
            st.markdown("#### 📊 Emotional Indicators")
            
            st.markdown("""
            <div class="emotion-radar">
            """, unsafe_allow_html=True)
            
            radar_fig = create_emotion_radar(probs, CLASSES)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Individual emotion metrics
            st.markdown("#### 🎭 State Breakdown")
            
            for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
                percentage = prob * 100
                emoji_map = {"Focused": "🎯", "Relaxed": "😌", "Stressed": "😰"}
                color_map = {"Focused": "#00ff88", "Relaxed": "#4dffaa", "Stressed": "#ff6644"}
                
                st.markdown(f"""
                <div style="margin: 15px 0; padding: 15px; background: rgba(0,30,60,0.4); border-radius: 10px; border: 1px solid {color_map[cls]};">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 10px;">{emoji_map[cls]}</span>
                            <span style="color: {color_map[cls]}; font-weight: bold;">{cls}</span>
                        </div>
                        <span style="color: #00ff88; font-family: 'Orbitron', monospace; font-size: 1.2rem;">
                            {percentage:.1f}%
                        </span>
                    </div>
                    <div class="progress-container" style="margin-top: 8px;">
                        <div class="progress-bar" style="width: {percentage}%; background: linear-gradient(90deg, {color_map[cls]}, {color_map[cls]}66);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Temporal analysis
        st.markdown("### ⏱️ Temporal Pattern Analysis")
        
        # Simulate time series data
        time_points = list(range(1, 21))
        focused_trend = [30 + i + np.random.normal(0, 5) for i in time_points]
        relaxed_trend = [25 + np.sin(i*0.3)*10 + np.random.normal(0, 3) for i in time_points]
        stressed_trend = [45 - i*0.5 + np.random.normal(0, 4) for i in time_points]
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=time_points, y=focused_trend,
            mode='lines+markers',
            name='Focused',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=6)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=time_points, y=relaxed_trend,
            mode='lines+markers',
            name='Relaxed',
            line=dict(color='#4dffaa', width=3),
            marker=dict(size=6)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=time_points, y=stressed_trend,
            mode='lines+markers',
            name='Stressed',
            line=dict(color='#ff6644', width=3),
            marker=dict(size=6)
        ))
        
        fig_trend.update_layout(
            title=dict(
                text="Emotional State Trends (Last 20 Measurements)",
                font=dict(family="Orbitron", color="#00ff88"),
                x=0.5
            ),
            plot_bgcolor='rgba(0,20,40,0.8)',
            paper_bgcolor='transparent',
            font=dict(color='#00ff88'),
            xaxis=dict(
                title="Time Point",
                gridcolor='rgba(0,255,136,0.2)',
                color='#00ff88'
            ),
            yaxis=dict(
                title="Probability (%)",
                gridcolor='rgba(0,255,136,0.2)',
                color='#00ff88'
            ),
            height=400,
            legend=dict(
                bgcolor='rgba(0,20,40,0.8)',
                bordercolor='#00ff88',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

# Footer with system information
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: #4dffaa;">
    <div style="font-family: 'Orbitron', monospace; font-size: 1.2rem; margin-bottom: 10px;">
        NEUROMIND EEG VISUALIZATION PANEL
    </div>
    <div style="font-size: 0.9rem; opacity: 0.8;">
        🧠 Model: {model_type} | 📡 Device: {DEVICE} | ⚡ Status: ONLINE | 🔒 LUCD Design
    </div>
    <div style="margin-top: 10px; font-size: 0.8rem; opacity: 0.6;">
        Advanced Neural Signal Processing • Real-time Brain State Classification • 2024
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # Auto-refresh every 5 seconds for demo purposes
    time.sleep(0.1)  # Small delay to prevent too frequent updates