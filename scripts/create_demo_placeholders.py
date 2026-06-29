#!/usr/bin/env python3
"""
Create placeholder demo images for NeuroMind screenshots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np

# Set professional styling
plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.facecolor': '#161b22',
    'figure.facecolor': '#0d1117',
    'text.color': '#f0f6fc',
    'axes.labelcolor': '#f0f6fc',
    'xtick.color': '#f0f6fc',
    'ytick.color': '#f0f6fc'
})

def create_main_dashboard_placeholder():
    """Create main dashboard placeholder."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('🧠 NeuroMind - EEG Brain Signal Classifier', fontsize=20, fontweight='bold')
    
    # EEG Signal Plot
    x = np.linspace(0, 4, 400)
    eeg_signal = np.sin(2*np.pi*10*x) + 0.3*np.sin(2*np.pi*25*x) + 0.1*np.random.randn(400)
    ax1.plot(x, eeg_signal, color='#00d4aa', linewidth=2)
    ax1.set_title('📡 Real-time EEG Signal (64 channels)', fontsize=14)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.grid(True, alpha=0.3)
    
    # Spectrogram
    frequencies = np.logspace(0.5, 1.7, 50)
    times = np.linspace(0, 4, 100)
    spectrogram = np.random.exponential(0.5, (50, 100))
    im = ax2.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower')
    ax2.set_title('🎵 STFT Spectrogram (4-45 Hz)', fontsize=14)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency (Hz)')
    
    # Prediction Results
    classes = ['🎯 Focused', '😌 Relaxed', '😰 Stressed']
    probabilities = [0.73, 0.18, 0.09]
    colors = ['#00d4aa', '#0969da', '#f85149']
    bars = ax3.bar(classes, probabilities, color=colors, alpha=0.8)
    ax3.set_title('🤖 AI Prediction Results', fontsize=14)
    ax3.set_ylabel('Confidence (%)')
    ax3.set_ylim(0, 1)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Model Performance Metrics
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    values = [67.3, 64.2, 69.1, 71.4]
    ax4.barh(metrics, values, color='#8957e5', alpha=0.8)
    ax4.set_title('📊 Model Performance Metrics', fontsize=14)
    ax4.set_xlabel('Score (%)')
    ax4.set_xlim(0, 100)
    
    # Add value labels
    for i, v in enumerate(values):
        ax4.text(v + 1, i, f'{v}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save placeholder
    output_dir = Path("assets/demo_screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "main_dashboard_placeholder.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created main dashboard placeholder")

def create_prediction_results_placeholder():
    """Create prediction results placeholder."""
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 2, 1], width_ratios=[2, 1, 2])
    
    # Main prediction display
    ax_main = fig.add_subplot(gs[1, :])
    ax_main.text(0.5, 0.7, '🎯 FOCUSED', fontsize=48, ha='center', va='center', 
                color='#00d4aa', fontweight='bold', transform=ax_main.transAxes)
    ax_main.text(0.5, 0.5, 'Mental State Detected', fontsize=16, ha='center', va='center',
                color='#f0f6fc', transform=ax_main.transAxes)
    ax_main.text(0.5, 0.3, '73.2% Confidence', fontsize=24, ha='center', va='center',
                color='#00d4aa', fontweight='bold', transform=ax_main.transAxes)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Add border
    rect = patches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, 
                           edgecolor='#00d4aa', facecolor='none', 
                           transform=ax_main.transAxes)
    ax_main.add_patch(rect)
    
    # Confidence bars
    ax_conf = fig.add_subplot(gs[0, :])
    classes = ['Focused', 'Relaxed', 'Stressed']
    confidences = [73.2, 18.5, 8.3]
    colors = ['#00d4aa', '#0969da', '#f85149']
    
    bars = ax_conf.barh(classes, confidences, color=colors, alpha=0.8)
    ax_conf.set_xlabel('Confidence (%)')
    ax_conf.set_title('🧠 Classification Confidence Breakdown', fontsize=14, fontweight='bold')
    ax_conf.set_xlim(0, 100)
    
    # Add percentage labels
    for bar, conf in zip(bars, confidences):
        width = bar.get_width()
        ax_conf.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{conf}%', ha='left', va='center', fontweight='bold')
    
    # Processing info
    ax_info = fig.add_subplot(gs[2, :])
    info_text = [
        "📊 Processing Time: 1.35s",
        "🧮 Model: Ensemble (ResNet18 + EfficientNet-B0)",
        "📡 Input: 64-channel EEG, 4-second epoch",
        "🎯 Accuracy: 67.3% on PhysioNet dataset"
    ]
    
    for i, info in enumerate(info_text):
        ax_info.text(0.25 * i, 0.5, info, transform=ax_info.transAxes,
                    fontsize=11, ha='left', va='center', color='#7d8590')
    
    ax_info.axis('off')
    
    plt.suptitle('🤖 NeuroMind AI Prediction Results', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save placeholder
    output_dir = Path("assets/demo_screenshots")
    plt.savefig(output_dir / "prediction_results_placeholder.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created prediction results placeholder")

def create_gradcam_placeholder():
    """Create Grad-CAM visualization placeholder."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original spectrogram
    np.random.seed(42)
    spectrogram = np.random.exponential(0.5, (128, 128))
    ax1.imshow(spectrogram, cmap='viridis', aspect='auto')
    ax1.set_title('📊 Original Spectrogram', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency (Hz)')
    
    # Grad-CAM heatmap
    heatmap = np.random.exponential(1.0, (128, 128))
    heatmap = np.exp(-((np.arange(128)[:, None] - 64)**2 + (np.arange(128)[None, :] - 64)**2) / 1000)
    ax2.imshow(heatmap, cmap='jet', aspect='auto')
    ax2.set_title('🔥 Grad-CAM Attention Map', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency (Hz)')
    
    # Overlay
    overlay = 0.7 * spectrogram + 0.3 * heatmap
    ax3.imshow(overlay, cmap='hot', aspect='auto')
    ax3.set_title('🧠 AI Attention Overlay', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency (Hz)')
    
    plt.suptitle('🔍 Explainable AI: Grad-CAM Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Add explanation text
    fig.text(0.5, 0.02, 
            '🎯 Red/Yellow areas show high AI attention • Blue areas show low attention • Reveals decision-making process',
            ha='center', fontsize=11, color='#7d8590', style='italic')
    
    # Save placeholder
    output_dir = Path("assets/demo_screenshots")
    plt.savefig(output_dir / "gradcam_interface_placeholder.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created Grad-CAM placeholder")

def main():
    """Generate all placeholder images."""
    print("🎨 Creating demo placeholder images...")
    
    create_main_dashboard_placeholder()
    create_prediction_results_placeholder()
    create_gradcam_placeholder()
    
    print("\n🎉 All placeholder images created!")
    print("📁 Location: assets/demo_screenshots/")
    print("\n💡 These are placeholder visualizations.")
    print("🚀 Run 'streamlit run src/app.py' to see the real interface!")

if __name__ == "__main__":
    main()