"""
Multimodal Emotion Recognition - Premium Demo
Clean, minimal, professional UI following OpenAI × Apple × Linear.app aesthetic
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import inference modules
from inference.speech_inference import load_speech_model, predict_speech_emotion
from inference.text_inference import load_text_model, predict_text_emotion
from inference.fusion_inference import load_fusion_model, predict_fusion_emotion

from templates import HERO_SECTION, SPEECH_INSIGHT, TEXT_INSIGHT, FUSION_INSIGHT, ARCHITECTURE_SECTION, FOOTER, prediction_card

warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PREMIUM MINIMAL CSS
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("style.css not found.")

# LOAD MODELS
@st.cache_resource
def load_models():
    """Load all three models"""
    root_dir = Path(__file__).parent.parent
    speech_model = load_speech_model(str(root_dir / "saved_models/advanced_speech_emotion_model.pth"))
    text_model = load_text_model(str(root_dir / "saved_models/text_emotion_model.pth"))
    fusion_model = load_fusion_model(str(root_dir / "saved_models/multimodal_fusion_model.pth"))
    return speech_model, text_model, fusion_model

try:
    speech_model, text_model, fusion_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}")

# EMOTION MAPPING
EMOTION_EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "pleasant_surprise": "😲",
    "sad": "😔"
}

# HERO SECTION
st.markdown(HERO_SECTION, unsafe_allow_html=True)

# INPUT SECTION
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("#### 📥 Input")

col1, col2 = st.columns(2)

with col1:
    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav"],
        help="Upload a WAV audio file for emotion analysis"
    )

with col2:
    text_input = st.text_area(
        "Text Transcript",
        placeholder="Enter the spoken word or transcript...",
        height=100,
        help="Enter the text content for emotion analysis"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ANALYZE BUTTON
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Emotion", use_container_width=True)

# ACOUSTIC ANALYTICS
if audio_file is not None:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Acoustic Analytics")
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=16000)
    duration = len(y) / sr
    
    # Metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Sample Rate", f"{sr} Hz")
    
    with metric_col2:
        st.metric("Duration", f"{duration:.2f}s")
    
    with metric_col3:
        rms = np.mean(librosa.feature.rms(y=y))
        st.metric("RMS Energy", f"{rms:.4f}")
    
    with metric_col4:
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        st.metric("Zero Crossing", f"{zcr:.4f}")
    
    # Waveform visualization
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#F59E0B', alpha=0.8)
    ax.set_facecolor('none')
    ax.set_xlabel('Time (s)', color='#94A3B8', fontsize=10)
    ax.set_ylabel('Amplitude', color='#94A3B8', fontsize=10)
    ax.tick_params(colors='#94A3B8')
    ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.1))
    ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('none')
    st.pyplot(fig)
    plt.close()
    
    st.markdown("</div>", unsafe_allow_html=True)

# PREDICTION RESULTS
if analyze_button and models_loaded:
    speech_result = None
    text_result = None
    fusion_result = None
    
    # Speech prediction
    if audio_file is not None:
        with st.spinner("Analyzing speech..."):
            try:
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_path = tmp_file.name
                
                speech_result = predict_speech_emotion(tmp_path, speech_model)
                speech_result["confidence"] = min(speech_result["confidence"], 0.998)
                if isinstance(speech_result.get("probabilities"), np.ndarray):
                    speech_result["probabilities"] = np.clip(speech_result["probabilities"], 0, 0.998)
                
                # Clean up temp file
                import os
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Speech analysis error: {e}")
    
    # Text prediction
    if text_input.strip():
        with st.spinner("Analyzing text..."):
            try:
                text_result = predict_text_emotion(text_input, text_model)
            except Exception as e:
                st.error(f"Text analysis error: {e}")
    
    # Fusion prediction
    if audio_file is not None and text_input.strip():
        with st.spinner("Performing multimodal fusion..."):
            try:
                # Save uploaded file temporarily
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_path = tmp_file.name
                
                fusion_result = predict_fusion_emotion(tmp_path, text_input, fusion_model)
                fusion_result["confidence"] = min(fusion_result["confidence"], 0.999)
                if isinstance(fusion_result.get("probabilities"), np.ndarray):
                    fusion_result["probabilities"] = np.clip(fusion_result["probabilities"], 0, 0.999)
                
                # Clean up temp file
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Fusion analysis error: {e}")
    
    # Display results
    if speech_result or text_result or fusion_result:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Predictions")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        # Speech prediction card
        with result_col1:
            if speech_result:
                emotion = speech_result["emotion"]
                confidence = speech_result["confidence"]
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                
                st.markdown(prediction_card(emoji, emotion, confidence, "Speech Model"), unsafe_allow_html=True)
            else:
                st.info("No audio provided")
        
        # Text prediction card
        with result_col2:
            if text_result:
                emotion = text_result["emotion"]
                confidence = text_result["confidence"]
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                
                st.markdown(prediction_card(emoji, emotion, confidence, "Text Model"), unsafe_allow_html=True)
            else:
                st.info("No text provided")
        
        # Fusion prediction card
        with result_col3:
            if fusion_result:
                emotion = fusion_result["emotion"]
                confidence = fusion_result["confidence"]
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                
                st.markdown(prediction_card(emoji, emotion, confidence, "Fusion Model", is_fusion=True), unsafe_allow_html=True)
            else:
                st.info("Provide both audio and text")
        
        # Confidence distribution
        if fusion_result or speech_result or text_result:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Confidence Distribution")
            
            # Use fusion if available, otherwise speech, otherwise text
            result_to_show = fusion_result or speech_result or text_result
            
            if result_to_show and "probabilities" in result_to_show:
                probs = result_to_show["probabilities"]
                
                # Define emotion labels
                emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
                
                # Handle numpy array
                if isinstance(probs, np.ndarray):
                    probs = probs.flatten()
                    for i, emotion in enumerate(emotion_labels):
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            st.write(f"{emotion.replace('_', ' ').title()}")
                        with col_b:
                            st.progress(float(probs[i]))
                            st.caption(f"{probs[i]:.1%}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Model Insights
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 💡 Model Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown(SPEECH_INSIGHT, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown(TEXT_INSIGHT, unsafe_allow_html=True)
        
        with insight_col3:
            st.markdown(FUSION_INSIGHT, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ARCHITECTURE SECTION
with st.expander("🏗️ System Architecture"):
    st.markdown(ARCHITECTURE_SECTION, unsafe_allow_html=True)

# FOOTER
st.markdown(FOOTER, unsafe_allow_html=True)
