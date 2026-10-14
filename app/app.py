"""
DroneSense: Intelligent Drone Detection Through Sound
Module: app/app.py
Author: Swati (24PD39)
Description: Full-Stack Interactive Streamlit Web Application for acoustic drone detection.
             Features:
             1. Live Detection (WAV upload, audio playback, Mel-spectrogram, Confidence Meter)
             2. Noise Simulator (Real-time SNR slider + AWGN noise injection)
             3. SHAP Explainability (Feature attribution breakdown)
             4. Benchmark Leaderboard (Multi-model metrics, ROC curves, Degradation charts)
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DroneSense: Acoustic Surveillance",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helper Functions: Audio DSP & Feature Extraction
# -----------------------------------------------------------------------------
def load_audio(file_bytes, target_sr=16000):
    """Load audio bytes, convert to mono and resample to 16 kHz."""
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=target_sr, mono=True)
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)
    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr


def extract_quick_features(y, sr=16000):
    """Extract standard 32 acoustic features from normalized audio signal."""
    # 13 MFCC means and std devs (26 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral features
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))

    features = np.hstack([mfcc_mean, mfcc_std, [centroid, bandwidth, rolloff, flatness, zcr, rms]])
    return features.reshape(1, -1)


def inject_noise_waveform(y, snr_db):
    """Inject AWGN directly into the audio waveform at target SNR level."""
    sig_power = np.mean(y ** 2)
    sig_power = max(sig_power, 1e-10)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=y.shape)
    noisy_y = y + noise
    # Normalize
    if np.max(np.abs(noisy_y)) > 0:
        noisy_y = noisy_y / np.max(np.abs(noisy_y))
    return noisy_y


def plot_mel_spectrogram(y, sr=16000, title="Mel-Spectrogram"):
    """Render a high-resolution Mel-spectrogram figure."""
    fig, ax = plt.subplots(figsize=(8, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    return fig


@st.cache_resource
def load_model():
    """Load trained model artifact if available, else load fallback classifier."""
    model_path = "best_drone_detector.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path), True
    else:
        # Pretrained fallback classifier for testing before pipeline merge
        from sklearn.ensemble import RandomForestClassifier
        np.random.seed(42)
        X_mock = np.random.randn(300, 32)
        y_mock = (X_mock[:, 0] * 1.5 + X_mock[:, 28] * 1.8 > 1.0).astype(int)
        model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_mock, y_mock)
        return model, False


model, is_live_model = load_model()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/drone.png", width=70)
st.sidebar.title("DroneSense Navigation")
module = st.sidebar.radio(
    "Select System Module:",
    ["1. Live Drone Detection", "2. Noise Simulator", "3. SHAP Explainability", "4. Benchmark Leaderboard"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**System Status**: {'Active (.joblib)' if is_live_model else 'Simulation Mode (RFC)'}\n\n"
    "**Team Ownership**:\n"
    "- Jaanu (24PD19): Audio DSP & Models\n"
    "- Swati (24PD39): Eval, XAI & App"
)

# -----------------------------------------------------------------------------
# MODULE 1: Live Drone Detection
# -----------------------------------------------------------------------------
if module == "1. Live Drone Detection":
    st.markdown("<div class='main-title'>🛸 Live Drone Acoustic Surveillance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Upload an audio clip (.wav) to detect incoming drone acoustic signatures in real-time.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Audio Sample (.wav)", type=["wav"])

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        y, sr = load_audio(file_bytes)

        with col1:
            st.subheader("🎧 Audio Playback & Analysis")
            st.audio(file_bytes, format="audio/wav")
            st.info(f"Sample Rate: **{sr} Hz** | Duration: **{len(y)/sr:.2f} s** | Channels: **Mono**")

            # Extract features & predict
            features = extract_quick_features(y, sr)
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(features)[0, 1])
            else:
                prob = float(model.predict(features)[0])

            confidence_pct = prob * 100

            st.subheader("🚨 Threat Assessment")
            if prob >= 0.5:
                st.error(f"⚠️ **THREAT DETECTED: Drone Acoustic Signature Identified!**")
            else:
                st.success(f"✅ **NO THREAT: Environmental Background Sound**")

            st.write(f"Detection Confidence: **{confidence_pct:.1f}%**")
            st.progress(prob)

        with col2:
            st.subheader("📊 Dynamic Mel-Spectrogram")
            fig = plot_mel_spectrogram(y, sr, title=f"Acoustic Signature: {uploaded_file.name}")
            st.pyplot(fig)
    else:
        st.info("👆 Upload a .wav audio clip above to begin analysis, or check the other modules.")

# -----------------------------------------------------------------------------
# MODULE 2: Environmental Noise Simulator
# -----------------------------------------------------------------------------
elif module == "2. Noise Simulator":
    st.markdown("<div class='main-title'>🌪️ Environmental Noise Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Simulate real-world outdoor noise interference (AWGN) and observe model stability.</div>", unsafe_allow_html=True)

    test_file = st.file_uploader("Upload Audio Clip for Noise Stress-Testing (.wav)", type=["wav"], key="noise_uploader")

    snr_slider = st.slider(
        "Simulated Signal-to-Noise Ratio (SNR in dB):",
        min_value=-5.0,
        max_value=20.0,
        value=15.0,
        step=1.0,
        help="Higher SNR (+20dB) = Clear Audio. Lower SNR (-5dB) = Heavy Background Noise.",
    )

    if test_file is not None:
        y_clean, sr = load_audio(test_file.read())
        y_noisy = inject_noise_waveform(y_clean, snr_db=snr_slider)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"🔊 Noisy Audio at {snr_slider:+.0f} dB SNR")
            # Convert numpy array to wav bytes for playback
            buffer = io.BytesIO()
            sf.write(buffer, y_noisy, sr, format="WAV")
            st.audio(buffer.getvalue(), format="audio/wav")

            # Predict on noisy audio
            noisy_feats = extract_quick_features(y_noisy, sr)
            noisy_prob = float(model.predict_proba(noisy_feats)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(noisy_feats)[0])

            st.metric("Drone Confidence (Under Noise)", f"{noisy_prob*100:.1f}%")
            if noisy_prob >= 0.5:
                st.error("Drone Signature Detected Despite Noise")
            else:
                st.warning("Signal Degraded / Classified as Environmental Noise")

        with col2:
            st.subheader("Spectrogram: Clean vs Noisy")
            fig_noisy = plot_mel_spectrogram(y_noisy, sr, title=f"Degraded Audio @ {snr_slider:+.0f} dB SNR")
            st.pyplot(fig_noisy)
    else:
        st.info("Upload an audio clip to inject live noise and hear the difference!")

# -----------------------------------------------------------------------------
# MODULE 3: SHAP Explainability
# -----------------------------------------------------------------------------
elif module == "3. SHAP Explainability":
    st.markdown("<div class='main-title'>🔍 Explainable AI: SHAP Interpretation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Understand which acoustic features (harmonic frequencies, roll-off) drive drone detection.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Global Acoustic Drivers (Beeswarm)")
        st.caption("Shows how high/low values of each frequency band impact classification across all samples.")
        beeswarm_img = "reports/shap_global_beeswarm.png"
        if os.path.exists(beeswarm_img):
            st.image(beeswarm_img, use_container_width=True)
        else:
            st.warning("Run `python src/explainability.py` to generate the global beeswarm plot.")

    with col2:
        st.subheader("🎯 Local Decision Breakdown (Waterfall)")
        st.caption("Step-by-step feature push for an individual audio sample.")
        waterfall_img = "reports/shap_local_waterfall_sample_0.png"
        if os.path.exists(waterfall_img):
            st.image(waterfall_img, use_container_width=True)
        else:
            st.warning("Run `python src/explainability.py` to generate the sample waterfall plot.")

    # Feature Importance Table
    importance_csv = "reports/shap_feature_importance.csv"
    if os.path.exists(importance_csv):
        st.subheader("📋 Top Driving Acoustic Indicators")
        df_imp = pd.read_csv(importance_csv)
        st.dataframe(df_imp.head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE 4: Benchmark Leaderboard
# -----------------------------------------------------------------------------
elif module == "4. Benchmark Leaderboard":
    st.markdown("<div class='main-title'>🏆 Multi-Model Benchmark Leaderboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Comparative performance across 6 classification algorithms and noise degradation.</div>", unsafe_allow_html=True)

    leaderboard_csv = "reports/model_benchmark.csv"
    if os.path.exists(leaderboard_csv):
        df_lead = pd.read_csv(leaderboard_csv)
        st.subheader("📊 Model Performance Table")
        st.dataframe(
            df_lead.style.highlight_max(
                subset=["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
                color="#C7D2FE",
            ),
            use_container_width=True,
        )
    else:
        st.info("Run `python src/evaluate.py` to generate the leaderboard table.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Comparative ROC Curves")
        roc_img = "reports/comparative_roc_curves.png"
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True)

    with col2:
        st.subheader("📉 Environmental Noise Degradation")
        deg_img = "reports/robustness_snr_degradation.png"
        if os.path.exists(deg_img):
            st.image(deg_img, use_container_width=True)