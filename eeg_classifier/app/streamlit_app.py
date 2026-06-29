"""
streamlit_app.py — Task 6
--------------------------
Professional EEG Brain Signal Classifier Demo.
Run: streamlit run app/streamlit_app.py

Features:
- Model selector: ResNet18 / EfficientNet-B0 / Ensemble
- Raw EEG → Spectrogram → Grad-CAM side by side
- Calibrated confidence with color coding
- Batch prediction mode
- Metrics dashboard tab
- Dark theme
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

from src.preprocess  import preprocess_raw, epoch_raw, epoch_to_spectrogram, load_physionet_data
from src.model       import build_model, get_last_conv_layer, build_ensemble
from src.gradcam     import GradCAM, overlay_heatmap
from src.model       import get_last_conv_layer
from src.calibration import load_temperature

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES   = ["Focused", "Relaxed", "Stressed"]
EMOJI     = {"Focused": "🎯", "Relaxed": "😌", "Stressed": "😰"}
COLORS    = {"Focused": "#00C853", "Relaxed": "#2979FF", "Stressed": "#FF1744"}
CKPT_DIR  = Path("models/checkpoints")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Brain Signal Classifier",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0F1117; }
.metric-card {
    background: linear-gradient(135deg,#1E1E2E,#2A2A3E);
    border-radius:14px; padding:18px; border:1px solid #3A3A5C;
    text-align:center; margin:5px 0;
}
.pred-card {
    border-radius:14px; padding:22px; text-align:center; margin:8px 0;
}
.section-title {
    font-size:17px; font-weight:700; color:#E0E0E0;
    margin-bottom:10px; padding-bottom:5px;
    border-bottom:2px solid #3A3A5C;
}
.info-box {
    background:#1E1E2E; border-left:4px solid #7C4DFF;
    border-radius:8px; padding:10px 14px; margin:6px 0;
    color:#B0B0C0; font-size:13px;
}
#MainMenu,footer,header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(arch):
    model = build_model(arch=arch, pretrained=False)
    ckpt  = CKPT_DIR / f"best_{arch}.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model.to(DEVICE).eval(), True
    # Fallback: try old path
    old = Path("models/best_model.pth")
    if old.exists():
        model.load_state_dict(torch.load(old, map_location=DEVICE))
        return model.to(DEVICE).eval(), True
    return model.to(DEVICE).eval(), False


@st.cache_resource(show_spinner=False)
def load_ensemble_cached():
    try:
        return build_ensemble(DEVICE), True
    except Exception as e:
        return None, False


# ── Helpers ────────────────────────────────────────────────────────────────────
def plot_eeg(raw, n_ch=6, duration=4):
    sfreq   = raw.info["sfreq"]
    n_samp  = int(sfreq * duration)
    data, t = raw[:n_ch, :n_samp]
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 4.5), sharex=True)
    fig.patch.set_facecolor('#0F1117')
    palette = ['#00BCD4','#4CAF50','#FF9800','#E91E63','#9C27B0','#2196F3']
    for i, ax in enumerate(axes):
        ax.plot(t, data[i]*1e6, lw=0.8, color=palette[i % len(palette)])
        ax.set_ylabel(f"Ch{i+1}", fontsize=7, color='#888')
        ax.set_facecolor('#0F1117')
        ax.tick_params(colors='#555', labelsize=6)
        ax.spines[:].set_color('#2A2A3E')
        ax.grid(True, alpha=0.12, color='#444')
    axes[-1].set_xlabel("Time (s)", color='#888', fontsize=8)
    plt.tight_layout(pad=0.4)
    return fig


def get_confidence_color(conf):
    """Task 6: Green >70%, Yellow 50-70%, Red <50%"""
    if conf >= 70:
        return "#00C853"
    elif conf >= 50:
        return "#FFD600"
    else:
        return "#FF1744"


def run_prediction(model, epoch_data, sfreq, temperature=1.0):
    img_np  = epoch_to_spectrogram(epoch_data, sfreq)
    img_pil = Image.fromarray(img_np)
    tensor  = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits / temperature, dim=1).squeeze().cpu().numpy()
    idx  = probs.argmax()
    return img_np, tensor, CLASSES[idx], float(probs[idx])*100, probs


def run_gradcam(model, arch, tensor, img_np):
    if arch == "ensemble":
        # Use ResNet18 branch for Grad-CAM in ensemble
        target_layer = get_last_conv_layer(model.resnet, "resnet18")
        gcam = GradCAM(model.resnet, target_layer)
    else:
        target_layer = get_last_conv_layer(model, arch)
        gcam = GradCAM(model, target_layer)

    heatmap = gcam(tensor.clone())
    gcam.remove_hooks()
    blended = overlay_heatmap(img_np, heatmap, alpha=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.patch.set_facecolor('#0F1117')
    for ax, im, title, cmap in zip(
        axes,
        [img_np, heatmap, blended],
        ["STFT Spectrogram", "Grad-CAM Heatmap", "Overlay (alpha=0.4)"],
        [None, "jet", None]
    ):
        ax.imshow(im, cmap=cmap, aspect='auto')
        ax.set_title(title, color='#CCC', fontsize=9, pad=5)
        ax.axis("off"); ax.set_facecolor('#0F1117')
    plt.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 EEG Classifier")
    st.divider()

    # Model selector (Task 6)
    arch_choice = st.selectbox(
        "Select Model",
        ["ResNet18", "EfficientNet-B0", "Ensemble (Both)"],
        index=0
    )
    arch_map = {
        "ResNet18":           "resnet18",
        "EfficientNet-B0":    "efficientnet_b0",
        "Ensemble (Both)":    "ensemble",
    }
    selected_arch = arch_map[arch_choice]

    st.divider()
    use_real     = st.toggle("Use Real PhysioNet EEG", value=True)
    epoch_idx    = st.slider("Epoch to analyse", 0, 15, 0)
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    use_calib    = st.checkbox("Use Calibrated Confidence", value=True)

    st.divider()
    st.markdown("""
    <div class='info-box'>
    <b>Dataset:</b> PhysioNet EEGBCI<br>
    <b>Subjects:</b> 109 | <b>Channels:</b> 64<br>
    <b>Filter:</b> 4-45 Hz STFT<br>
    <b>Labels:</b> Task-based mapping
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <b>Confidence Colors:</b><br>
    🟢 Green: > 70%<br>
    🟡 Yellow: 50-70%<br>
    🔴 Red: < 50%
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:8px 0 16px 0'>
    <h1 style='color:#E0E0FF;font-size:2.2em;margin:0'>
        🧠 EEG Brain Signal Classifier
    </h1>
    <p style='color:#888;font-size:1em;margin-top:5px'>
        ResNet18 + EfficientNet-B0 · STFT Spectrograms · Grad-CAM · PhysioNet Dataset
    </p>
    <div style='display:flex;justify-content:center;gap:16px;margin-top:8px'>
        <span style='background:#1E3A2E;color:#00C853;padding:3px 12px;border-radius:20px;font-size:12px'>🎯 Focused</span>
        <span style='background:#1A2A4A;color:#2979FF;padding:3px 12px;border-radius:20px;font-size:12px'>😌 Relaxed</span>
        <span style='background:#3A1A1A;color:#FF1744;padding:3px 12px;border-radius:20px;font-size:12px'>😰 Stressed</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "📦 Batch Prediction", "📊 Metrics Dashboard"])

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {arch_choice}..."):
    if selected_arch == "ensemble":
        model, loaded = load_ensemble_cached()
        if not loaded or model is None:
            st.warning("Ensemble requires both models trained. Using ResNet18.")
            model, loaded = load_model_cached("resnet18")
            selected_arch = "resnet18"
    else:
        model, loaded = load_model_cached(selected_arch)

temperature = load_temperature() if use_calib else 1.0

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Input
    if use_real:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            if st.button("🔄 Load New Subject", use_container_width=True):
                for k in ["eeg_raw","eeg_label"]:
                    st.session_state.pop(k, None)

        if "eeg_raw" not in st.session_state:
            with st.spinner("Downloading real EEG from PhysioNet..."):
                import random
                raw_list, labels = load_physionet_data(n_subjects=2, runs=[1,3,4])
                idx = random.randint(0, len(raw_list)-1)
                st.session_state["eeg_raw"]   = raw_list[idx]
                st.session_state["eeg_label"] = labels[idx]

        raw        = st.session_state["eeg_raw"]
        true_label = CLASSES[st.session_state["eeg_label"]]
        with col_info:
            st.markdown(f"""
            <div class='info-box'>
            📡 <b>PhysioNet EEGBCI</b> (real human EEG) &nbsp;|&nbsp;
            True label: <b>{EMOJI[true_label]} {true_label}</b> &nbsp;|&nbsp;
            64 channels · 160 Hz
            </div>""", unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader("Upload DEAP .dat file", type=["dat"])
        if not uploaded:
            st.info("Enable 'Use Real PhysioNet EEG' in sidebar or upload a DEAP .dat file.")
            st.stop()
        import pickle, mne, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
            tmp.write(uploaded.read())
        with open(tmp.name, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        eeg  = data['data'][0, :32, :]
        info = mne.create_info([f"EEG{i+1:02d}" for i in range(32)],
                               sfreq=128.0, ch_types=["eeg"]*32)
        raw  = mne.io.RawArray(eeg * 1e-6, info, verbose=False)
        true_label = "Unknown"

    # Preprocess
    with st.spinner("Preprocessing..."):
        raw_clean  = preprocess_raw(raw)
        epochs     = epoch_raw(raw_clean)
        epoch_data = epochs.get_data()
        sfreq      = raw.info["sfreq"]

    if len(epoch_data) == 0:
        st.error("No valid epochs. Try another signal.")
        st.stop()

    ep = epoch_data[min(epoch_idx, len(epoch_data)-1)]

    # EEG plot
    st.markdown("<div class='section-title'>📈 Raw EEG Signal (4-45 Hz filtered)</div>",
                unsafe_allow_html=True)
    st.pyplot(plot_eeg(raw_clean), use_container_width=True)
    plt.close()

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, val) in zip([m1,m2,m3,m4], [
        ("Channels", "64 EEG"),
        ("Sample Rate", "160 Hz"),
        ("Epochs", str(len(epoch_data))),
        ("Model", arch_choice),
    ]):
        col.markdown(f"""
        <div class='metric-card'>
            <div style='color:#888;font-size:11px'>{label}</div>
            <div style='color:#E0E0FF;font-size:18px;font-weight:700;margin-top:3px'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Prediction
    with st.spinner("Running prediction..."):
        img_np, tensor, pred_class, confidence, probs = run_prediction(
            model, ep, sfreq, temperature
        )

    st.markdown("<div class='section-title'>🔬 Spectrogram & Prediction</div>",
                unsafe_allow_html=True)

    col_spec, col_pred = st.columns([1, 1])

    with col_spec:
        fig_s, ax = plt.subplots(figsize=(5, 4))
        fig_s.patch.set_facecolor('#0F1117')
        ax.imshow(img_np, aspect='auto', origin='upper')
        ax.set_title("STFT Spectrogram (4-45 Hz)", color='#CCC', fontsize=10)
        ax.set_xlabel("Time", color='#888', fontsize=8)
        ax.set_ylabel("Frequency", color='#888', fontsize=8)
        ax.tick_params(colors='#555'); ax.set_facecolor('#0F1117')
        st.pyplot(fig_s, use_container_width=True); plt.close()

    with col_pred:
        conf_color = get_confidence_color(confidence)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1A1A2E,#2A2A3E);
                    border:2px solid {conf_color};border-radius:14px;
                    padding:24px;text-align:center;margin-bottom:14px'>
            <div style='font-size:48px'>{EMOJI[pred_class]}</div>
            <div style='color:{conf_color};font-size:28px;font-weight:800;
                        letter-spacing:2px;margin:6px 0'>{pred_class.upper()}</div>
            <div style='color:#AAA;font-size:13px'>Mental State Detected</div>
            <div style='background:{conf_color}22;border-radius:30px;
                        padding:6px 18px;display:inline-block;margin-top:10px'>
                <span style='color:{conf_color};font-size:20px;font-weight:700'>
                    {confidence:.1f}% {"(calibrated)" if use_calib and temperature != 1.0 else ""}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Probability Distribution**")
        for cls, prob in zip(CLASSES, probs):
            c   = COLORS[cls]
            pct = prob * 100
            st.markdown(f"""
            <div style='margin:5px 0'>
                <div style='display:flex;justify-content:space-between;
                            color:#CCC;font-size:12px;margin-bottom:2px'>
                    <span>{EMOJI[cls]} {cls}</span>
                    <span style='color:{c};font-weight:700'>{pct:.1f}%</span>
                </div>
                <div style='background:#2A2A3E;border-radius:6px;height:9px;overflow:hidden'>
                    <div style='background:{c};width:{pct}%;height:100%;border-radius:6px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        if true_label != "Unknown":
            match = "Correct" if pred_class == true_label else "Incorrect"
            mc    = "#00C853" if match == "Correct" else "#FF1744"
            st.markdown(f"""
            <div style='margin-top:12px;background:#1A1A2E;border-radius:8px;
                        padding:8px 14px;color:#AAA;font-size:12px'>
                True: <b style='color:#E0E0FF'>{EMOJI[true_label]} {true_label}</b>
                &nbsp; <span style='color:{mc}'>● {match}</span>
            </div>""", unsafe_allow_html=True)

    # Grad-CAM
    if show_gradcam:
        st.divider()
        st.markdown("<div class='section-title'>🔥 Grad-CAM Explainability</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        Highlights which <b>frequency bands and time regions</b> the CNN used.
        Red/yellow = high importance · Blue = low importance · alpha=0.4
        </div>""", unsafe_allow_html=True)
        with st.spinner("Generating Grad-CAM..."):
            fig_gc = run_gradcam(model, selected_arch, tensor, img_np)
        st.pyplot(fig_gc, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📦 Batch Prediction")
    st.markdown("Upload multiple spectrogram PNG images for batch classification.")

    uploaded_files = st.file_uploader(
        "Upload spectrogram images (.png)",
        type=["png"], accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        prog    = st.progress(0)
        for i, f in enumerate(uploaded_files):
            img_pil = Image.open(f).convert("RGB")
            img_np  = np.array(img_pil.resize((224, 224)))
            tensor  = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits / temperature, dim=1).squeeze().cpu().numpy()
            idx  = probs.argmax()
            results.append({
                "File":       f.name,
                "Prediction": CLASSES[idx],
                "Confidence": f"{probs[idx]*100:.1f}%",
                "Focused":    f"{probs[0]*100:.1f}%",
                "Relaxed":    f"{probs[1]*100:.1f}%",
                "Stressed":   f"{probs[2]*100:.1f}%",
            })
            prog.progress((i+1) / len(uploaded_files))

        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Summary
        from collections import Counter
        counts = Counter(r["Prediction"] for r in results)
        c1, c2, c3 = st.columns(3)
        for col, cls in zip([c1,c2,c3], CLASSES):
            col.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:24px'>{EMOJI[cls]}</div>
                <div style='color:{COLORS[cls]};font-size:22px;font-weight:700'>
                    {counts.get(cls, 0)}
                </div>
                <div style='color:#888;font-size:11px'>{cls}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Upload PNG spectrogram images above to run batch prediction.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — METRICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Metrics Dashboard")

    results_dir = Path("results")
    plots = {
        "Confusion Matrix":  f"confusion_matrix_{selected_arch}.png",
        "ROC Curves":        f"roc_curves_{selected_arch}.png",
        "Calibration Curve": f"calibration_{selected_arch}.png",
        "Training Curves":   "training_curves.png",
        "Model Comparison":  "model_comparison.png",
    }

    found_any = False
    cols = st.columns(2)
    col_idx = 0
    for title, fname in plots.items():
        path = results_dir / fname
        if not path.exists():
            path = Path("models") / fname
        if path.exists():
            with cols[col_idx % 2]:
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)
            col_idx += 1
            found_any = True

    if not found_any:
        st.info("No result plots found yet. Run `python download_data.py` first to train and evaluate.")

    # Grad-CAM samples
    gcam_dir = Path("results/gradcam")
    gcam_imgs = list(gcam_dir.glob("*.png")) if gcam_dir.exists() else []
    if gcam_imgs:
        st.markdown("---")
        st.markdown("**Grad-CAM Samples (all classes)**")
        gcols = st.columns(len(gcam_imgs))
        for col, img_path in zip(gcols, gcam_imgs):
            col.image(str(img_path), caption=img_path.stem,
                      use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#444;font-size:11px;padding:8px'>
    EEG Brain Signal Classifier &nbsp;|&nbsp; ResNet18 + EfficientNet-B0 &nbsp;|&nbsp;
    PhysioNet EEGBCI &nbsp;|&nbsp; Grad-CAM &nbsp;|&nbsp; PyTorch + Streamlit
</div>""", unsafe_allow_html=True)
