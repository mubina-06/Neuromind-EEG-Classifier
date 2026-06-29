"""
app.py — EEG Brain Signal Classifier
Professional Streamlit Demo App
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from pathlib import Path
import time

from src.preprocess import preprocess_raw, epoch_raw, epoch_to_spectrogram, load_physionet_data
from src.model      import build_model, get_last_conv_layer
from src.gradcam    import GradCAM

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES     = ["Focused", "Relaxed", "Stressed"]
EMOJI       = {"Focused": "🎯", "Relaxed": "😌", "Stressed": "😰"}
COLORS      = {"Focused": "#00C853", "Relaxed": "#2979FF", "Stressed": "#FF1744"}
BG_COLORS   = {"Focused": "#E8F5E9", "Relaxed": "#E3F2FD", "Stressed": "#FFEBEE"}
MODEL_PATH  = "models/best_model.pth"
ARCH        = "resnet18"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Brain Signal Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0F1117; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E1E2E, #2A2A3E);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #3A3A5C;
        text-align: center;
        margin: 6px 0;
    }
    .result-card {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 10px 0;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #E0E0E0;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #3A3A5C;
    }
    .info-box {
        background: #1E1E2E;
        border-left: 4px solid #7C4DFF;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #B0B0C0;
        font-size: 14px;
    }
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = build_model(arch=ARCH, pretrained=False)
    if Path(MODEL_PATH).exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


# ── Helpers ────────────────────────────────────────────────────────────────────
def plot_eeg(raw, n_ch=6, duration=4):
    sfreq    = raw.info["sfreq"]
    n_samp   = int(sfreq * duration)
    data, t  = raw[:n_ch, :n_samp]
    fig, axes = plt.subplots(n_ch, 1, figsize=(11, 5), sharex=True)
    fig.patch.set_facecolor('#0F1117')
    ch_colors = ['#00BCD4','#4CAF50','#FF9800','#E91E63','#9C27B0','#2196F3']
    for i, ax in enumerate(axes):
        ax.plot(t, data[i]*1e6, linewidth=0.9, color=ch_colors[i % len(ch_colors)])
        ax.set_ylabel(f"Ch{i+1}", fontsize=7, color='#888')
        ax.set_facecolor('#0F1117')
        ax.tick_params(colors='#555', labelsize=6)
        ax.spines[:].set_color('#2A2A3E')
        ax.grid(True, alpha=0.15, color='#444')
    axes[-1].set_xlabel("Time (s)", color='#888', fontsize=8)
    plt.tight_layout(pad=0.5)
    return fig


def predict(model, epoch_data, sfreq):
    img_np  = epoch_to_spectrogram(epoch_data, sfreq)
    img_pil = Image.fromarray(img_np)
    tensor  = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    idx   = probs.argmax()
    return img_np, tensor, CLASSES[idx], float(probs[idx])*100, probs


def run_gradcam(model, tensor, img_np):
    target_layer = get_last_conv_layer(model, ARCH)
    gcam         = GradCAM(model, target_layer)
    heatmap      = gcam(tensor.clone())

    import matplotlib.cm as cm_mod
    H, W = img_np.shape[:2]
    hm   = np.array(Image.fromarray((heatmap*255).astype(np.uint8)).resize((W,H), Image.LANCZOS)) / 255.0
    rgb  = (cm_mod.get_cmap("jet")(hm)[:,:,:3] * 255).astype(np.uint8)
    blend = (0.55*rgb + 0.45*img_np).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.patch.set_facecolor('#0F1117')
    titles = ["STFT Spectrogram", "Grad-CAM Heatmap", "Overlay"]
    imgs   = [img_np, heatmap, blend]
    cmaps  = [None, "jet", None]
    for ax, im, title, cmap in zip(axes, imgs, titles, cmaps):
        ax.imshow(im, cmap=cmap, aspect='auto')
        ax.set_title(title, color='#CCC', fontsize=10, pad=6)
        ax.axis("off")
        ax.set_facecolor('#0F1117')
    plt.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 EEG Classifier")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    use_mock    = st.toggle("Use Real PhysioNet EEG", value=True)
    epoch_idx   = st.slider("Epoch to analyse", 0, 15, 0)
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_metrics = st.checkbox("Show Signal Metrics", value=True)

    st.markdown("---")
    st.markdown("### 📋 Project Info")
    st.markdown("""
    <div class='info-box'>
    <b>Dataset:</b> DEAP<br>
    <b>Model:</b> ResNet18<br>
    <b>Filter:</b> 4–45 Hz<br>
    <b>Transform:</b> STFT<br>
    <b>Classes:</b> 3<br>
    <b>Labels:</b> Arousal-based
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏷️ Label Mapping")
    st.markdown("""
    <div class='info-box'>
    😰 <b>Stressed</b> → Arousal ≥ 6<br>
    🎯 <b>Focused</b>  → Arousal 4–5<br>
    😌 <b>Relaxed</b>  → Arousal ≤ 3
    </div>
    """, unsafe_allow_html=True)

    if Path(MODEL_PATH).exists():
        st.success("✅ Trained model loaded")
    else:
        st.warning("⚠️ No trained model found")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding: 10px 0 20px 0'>
    <h1 style='color:#E0E0FF; font-size:2.4em; margin:0'>
        🧠 EEG Brain Signal Classifier
    </h1>
    <p style='color:#888; font-size:1.05em; margin-top:6px'>
        Deep Learning · ResNet18 · STFT Spectrograms · Grad-CAM Explainability
    </p>
    <div style='display:flex; justify-content:center; gap:20px; margin-top:10px'>
        <span style='background:#1E3A2E; color:#00C853; padding:4px 14px; border-radius:20px; font-size:13px'>🎯 Focused</span>
        <span style='background:#1A2A4A; color:#2979FF; padding:4px 14px; border-radius:20px; font-size:13px'>😌 Relaxed</span>
        <span style='background:#3A1A1A; color:#FF1744; padding:4px 14px; border-radius:20px; font-size:13px'>😰 Stressed</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════
model = load_model()

if use_mock:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Load New Subject", use_container_width=True):
            st.session_state.pop("eeg_raw", None)
            st.session_state.pop("eeg_label", None)

    if "eeg_raw" not in st.session_state:
        with st.spinner("📡 Downloading real EEG from PhysioNet (first time ~10s)..."):
            import random
            subj = random.randint(1, 20)
            raw_list, labels = load_physionet_data(n_subjects=1, runs=[1, 3, 4])
            idx = random.randint(0, len(raw_list)-1)
            st.session_state["eeg_raw"]   = raw_list[idx]
            st.session_state["eeg_label"] = labels[idx]

    raw        = st.session_state["eeg_raw"]
    true_label = CLASSES[st.session_state["eeg_label"]]

    with col_info:
        st.markdown(f"""
        <div class='info-box'>
        📡 <b>Signal source:</b> PhysioNet EEGBCI (real EEG data) &nbsp;|&nbsp;
        🏷️ <b>True label:</b> {EMOJI[true_label]} <b>{true_label}</b> &nbsp;|&nbsp;
        📊 <b>Channels:</b> 64 &nbsp;|&nbsp; ⏱️ <b>Sampling rate:</b> 160 Hz
        </div>
        """, unsafe_allow_html=True)

else:
    uploaded = st.file_uploader(
        "📂 Upload DEAP .dat file (s01.dat ... s32.dat)",
        type=["dat"],
        help="Download from: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/"
    )
    if not uploaded:
        st.markdown("""
        <div style='background:#1A1A2E; border:2px dashed #3A3A6E; border-radius:12px;
                    padding:30px; text-align:center; color:#888; margin:20px 0'>
            <h3 style='color:#7C4DFF'>📥 No file uploaded</h3>
            <p>Upload a DEAP <b>.dat</b> file above, or enable <b>Mock EEG Signal</b> in the sidebar to run a demo.</p>
            <p style='font-size:12px'>DEAP download: <code>https://www.eecs.qmul.ac.uk/mmv/datasets/deap/</code></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    import pickle, mne, tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    with open(tmp_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    eeg  = data['data'][0, :32, :]
    info = mne.create_info([f"EEG{i+1:02d}" for i in range(32)],
                           sfreq=128.0, ch_types=["eeg"]*32)
    raw  = mne.io.RawArray(eeg * 1e-6, info, verbose=False)
    true_label = "Unknown"
    st.success(f"✅ File loaded: {uploaded.name}")

# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING + EEG PLOT
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Preprocessing EEG signal..."):
    raw_clean  = preprocess_raw(raw)
    epochs     = epoch_raw(raw_clean)
    epoch_data = epochs.get_data()
    sfreq      = raw.info["sfreq"]

if len(epoch_data) == 0:
    st.error("No valid epochs found. Try a different signal.")
    st.stop()

ep_idx = min(epoch_idx, len(epoch_data) - 1)
ep     = epoch_data[ep_idx]

st.markdown("<div class='section-title'>📈 Raw EEG Signal (after 4–45 Hz filtering)</div>",
            unsafe_allow_html=True)
fig_eeg = plot_eeg(raw_clean)
st.pyplot(fig_eeg, use_container_width=True)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL METRICS
# ══════════════════════════════════════════════════════════════════════════════
if show_metrics:
    data_vals = raw_clean.get_data() * 1e6
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("📊 Channels",    "32 EEG"),
        ("⏱️ Sample Rate", "128 Hz"),
        ("📏 Epochs",      str(len(epoch_data))),
        ("🔬 Epoch Length","4 seconds"),
    ]
    for col, (label, val) in zip([m1,m2,m3,m4], metrics):
        col.markdown(f"""
        <div class='metric-card'>
            <div style='color:#888; font-size:12px'>{label}</div>
            <div style='color:#E0E0FF; font-size:22px; font-weight:700; margin-top:4px'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Running CNN prediction..."):
    img_np, tensor, pred_class, confidence, probs = predict(model, ep, sfreq)

st.markdown("<div class='section-title'>🔬 STFT Spectrogram & Prediction</div>",
            unsafe_allow_html=True)

col_spec, col_result = st.columns([1, 1])

with col_spec:
    st.markdown(f"**Epoch {ep_idx} — STFT Spectrogram**")
    fig_spec, ax = plt.subplots(figsize=(6, 4))
    fig_spec.patch.set_facecolor('#0F1117')
    ax.imshow(img_np, aspect='auto', origin='upper')
    ax.set_title("Frequency vs Time (4–45 Hz)", color='#CCC', fontsize=10)
    ax.set_xlabel("Time →", color='#888', fontsize=9)
    ax.set_ylabel("Frequency →", color='#888', fontsize=9)
    ax.tick_params(colors='#555')
    ax.set_facecolor('#0F1117')
    st.pyplot(fig_spec, use_container_width=True)
    plt.close()

with col_result:
    color = COLORS[pred_class]
    bg    = BG_COLORS[pred_class]
    emoji = EMOJI[pred_class]

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1A1A2E,#2A2A3E);
                border:2px solid {color}; border-radius:16px;
                padding:28px; text-align:center; margin-bottom:16px'>
        <div style='font-size:52px'>{emoji}</div>
        <div style='color:{color}; font-size:32px; font-weight:800;
                    letter-spacing:2px; margin:8px 0'>{pred_class.upper()}</div>
        <div style='color:#AAA; font-size:15px'>Mental State Detected</div>
        <div style='background:{color}22; border-radius:30px; padding:8px 20px;
                    display:inline-block; margin-top:12px'>
            <span style='color:{color}; font-size:22px; font-weight:700'>
                {confidence:.1f}% Confidence
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Probability Distribution**")
    for cls, prob in zip(CLASSES, probs):
        c = COLORS[cls]
        pct = prob * 100
        st.markdown(f"""
        <div style='margin:6px 0'>
            <div style='display:flex; justify-content:space-between;
                        color:#CCC; font-size:13px; margin-bottom:3px'>
                <span>{EMOJI[cls]} {cls}</span>
                <span style='color:{c}; font-weight:700'>{pct:.1f}%</span>
            </div>
            <div style='background:#2A2A3E; border-radius:8px; height:10px; overflow:hidden'>
                <div style='background:{c}; width:{pct}%; height:100%;
                            border-radius:8px; transition:width 0.5s'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if true_label != "Unknown":
        match = "✅ Correct" if pred_class == true_label else "❌ Incorrect"
        st.markdown(f"""
        <div style='margin-top:14px; background:#1A1A2E; border-radius:10px;
                    padding:10px 16px; color:#AAA; font-size:13px'>
            True Label: <b style='color:#E0E0FF'>{EMOJI[true_label]} {true_label}</b>
            &nbsp;&nbsp; {match}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
if show_gradcam:
    st.divider()
    st.markdown("<div class='section-title'>🔥 Grad-CAM Explainability</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    Grad-CAM highlights <b>which frequency bands and time regions</b> the CNN focused on
    to make its prediction. Red/yellow = high importance, Blue = low importance.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating Grad-CAM heatmap..."):
        fig_gcam = run_gradcam(model, tensor, img_np)
    st.pyplot(fig_gcam, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📊 View Full Pipeline", expanded=False):
    st.markdown("""
    <div style='display:flex; flex-wrap:wrap; gap:8px; justify-content:center;
                padding:16px; background:#1A1A2E; border-radius:12px'>
    """, unsafe_allow_html=True)

    steps = [
        ("📥", "Load EEG", "DEAP .dat"),
        ("🔧", "Filter", "4–45 Hz"),
        ("✂️", "Epoch", "4s windows"),
        ("📊", "STFT", "Spectrogram"),
        ("🏷️", "Label", "Arousal map"),
        ("🧠", "ResNet18", "CNN model"),
        ("🎯", "Predict", "3 classes"),
        ("🔥", "Grad-CAM", "Explain"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, sub) in zip(cols, steps):
        col.markdown(f"""
        <div style='background:#2A2A3E; border-radius:10px; padding:10px 6px;
                    text-align:center; border:1px solid #3A3A5C'>
            <div style='font-size:22px'>{icon}</div>
            <div style='color:#E0E0FF; font-size:12px; font-weight:700'>{title}</div>
            <div style='color:#666; font-size:10px'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style='text-align:center; color:#444; font-size:12px; padding:10px'>
    EEG Brain Signal Classifier &nbsp;|&nbsp; ResNet18 + STFT &nbsp;|&nbsp;
    DEAP Dataset &nbsp;|&nbsp; Grad-CAM Explainability &nbsp;|&nbsp;
    Built with PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
