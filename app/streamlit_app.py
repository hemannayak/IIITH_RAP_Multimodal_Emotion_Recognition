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

# Import inference modules
from inference.speech_inference import load_speech_model, predict_speech_emotion
from inference.text_inference import load_text_model, predict_text_emotion
from inference.fusion_inference import load_fusion_model, predict_fusion_emotion

warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PREMIUM MINIMAL CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0E1117;
        color: #F8FAFC;
    }
    
    /* Hide Sidebar */
    [data-testid="stSidebar"] { display: none; }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Labels */
    label[data-testid="stWidgetLabel"] p {
        color: #F8FAFC !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #7C3AED !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(22, 27, 34, 0.6);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: rgba(22, 27, 34, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(124, 58, 237, 0.2);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .prediction-card:hover {
        border-color: rgba(124, 58, 237, 0.4);
        transform: translateY(-2px);
    }
    
    /* Badge */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #A78BFA;
        margin: 0 0.25rem;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.025em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.4) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7C3AED, #A78BFA);
        border-radius: 4px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(22, 27, 34, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Text Area */
    textarea {
        background: rgba(22, 27, 34, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Info Box */
    .info-card {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #64748B;
        font-size: 0.875rem;
    }
    
    .accent { color: #7C3AED; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    """Load all three models"""
    speech_model = load_speech_model("saved_models/advanced_speech_emotion_model.pth")
    text_model = load_text_model("saved_models/text_emotion_model.pth")
    fusion_model = load_fusion_model("saved_models/multimodal_fusion_model.pth")
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
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;'>
        🎙️ Multimodal Emotion Recognition
    </h1>
    <p style='color: #94A3B8; font-size: 1rem; margin-bottom: 1.5rem;'>
        Speech • Text • Multimodal Fusion
    </p>
    <div style='margin-bottom: 2rem;'>
        <span class='metric-badge'>Speech: 100%</span>
        <span class='metric-badge'>Text: 13.81%</span>
        <span class='metric-badge'>Fusion: 100%</span>
    </div>
</div>
""", unsafe_allow_html=True)

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
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#7C3AED', alpha=0.8)
    ax.set_facecolor('#0E1117')
    ax.set_xlabel('Time (s)', color='#94A3B8', fontsize=10)
    ax.set_ylabel('Amplitude', color='#94A3B8', fontsize=10)
    ax.tick_params(colors='#94A3B8')
    ax.spines['bottom'].set_color('#94A3B8')
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#0E1117')
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
                speech_result = predict_speech_emotion(audio_file, speech_model)
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
                fusion_result = predict_fusion_emotion(audio_file, text_input, fusion_model)
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
                
                st.markdown(f"""
                <div class='prediction-card'>
                    <p style='color: #94A3B8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Speech Model</p>
                    <div style='font-size: 3rem; margin: 0.5rem 0;'>{emoji}</div>
                    <h3 style='color: #F8FAFC; margin: 0.5rem 0; font-size: 1.25rem;'>{emotion.replace('_', ' ').title()}</h3>
                    <p style='color: #7C3AED; font-weight: 700; font-size: 1.5rem; margin-top: 0.5rem;'>{confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No audio provided")
        
        # Text prediction card
        with result_col2:
            if text_result:
                emotion = text_result["emotion"]
                confidence = text_result["confidence"]
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                
                st.markdown(f"""
                <div class='prediction-card'>
                    <p style='color: #94A3B8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Text Model</p>
                    <div style='font-size: 3rem; margin: 0.5rem 0;'>{emoji}</div>
                    <h3 style='color: #F8FAFC; margin: 0.5rem 0; font-size: 1.25rem;'>{emotion.replace('_', ' ').title()}</h3>
                    <p style='color: #7C3AED; font-weight: 700; font-size: 1.5rem; margin-top: 0.5rem;'>{confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No text provided")
        
        # Fusion prediction card
        with result_col3:
            if fusion_result:
                emotion = fusion_result["emotion"]
                confidence = fusion_result["confidence"]
                emoji = EMOTION_EMOJIS.get(emotion, "🎭")
                
                st.markdown(f"""
                <div class='prediction-card'>
                    <p style='color: #94A3B8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Fusion Model</p>
                    <div style='font-size: 3rem; margin: 0.5rem 0;'>{emoji}</div>
                    <h3 style='color: #F8FAFC; margin: 0.5rem 0; font-size: 1.25rem;'>{emotion.replace('_', ' ').title()}</h3>
                    <p style='color: #7C3AED; font-weight: 700; font-size: 1.5rem; margin-top: 0.5rem;'>{confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
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
                
                for emotion, prob in probs.items():
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        st.write(f"{emotion.replace('_', ' ').title()}")
                    with col_b:
                        st.progress(float(prob))
                        st.caption(f"{prob:.1%}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Model Insights
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 💡 Model Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown("""
            <div class='info-card' style='background: rgba(124, 58, 237, 0.1); border-left: 3px solid #7C3AED;'>
                <p style='font-weight: 600; margin-bottom: 0.5rem;'>Speech Dominant</p>
                <p style='font-size: 0.875rem; color: #94A3B8; margin: 0;'>Acoustic features provide robust emotional cues through prosody and spectral patterns.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown("""
            <div class='info-card' style='background: rgba(239, 68, 68, 0.1); border-left: 3px solid #EF4444;'>
                <p style='font-weight: 600; margin-bottom: 0.5rem;'>Text Limited</p>
                <p style='font-size: 0.875rem; color: #94A3B8; margin: 0;'>Isolated words lack emotional semantics. Performance improves with full sentences.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col3:
            st.markdown("""
            <div class='info-card' style='background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981;'>
                <p style='font-weight: 600; margin-bottom: 0.5rem;'>Fusion Integrates</p>
                <p style='font-size: 0.875rem; color: #94A3B8; margin: 0;'>Combines acoustic and semantic modalities for comprehensive emotion analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ARCHITECTURE SECTION
with st.expander("🏗️ System Architecture"):
    st.markdown("""
    ### Pipeline Overview
    
    **Speech Pipeline:**
    ```
    Audio → Trim Silence → Normalize → MFCC(40) + Δ + ΔΔ
    → CNN + BiLSTM + Attention → FC(128) → Softmax(7)
    ```
    
    **Text Pipeline:**
    ```
    Text → Contextual Prompt → DistilBERT → [CLS]
    → FC(256) → FC(128) → Softmax(7)
    ```
    
    **Fusion Pipeline:**
    ```
    Speech Encoder (128-dim) ⊕ Text Encoder (256-dim)
    → Fusion(384) → FC(256) → FC(128) → Softmax(7)
    ```
    
    ### Model Performance
    - **Speech:** 100% accuracy (near-perfect on held-out test set)
    - **Text:** 13.81% accuracy (limited by isolated word transcripts)
    - **Fusion:** 100% accuracy (speech-dominated multimodal integration)
    """)

# FOOTER
st.markdown("""
<div class='footer'>
    <p style='margin-bottom: 0.5rem;'>
        Built for <span class='accent'>IIITH Research Assistant Program</span>
    </p>
    <p style='font-size: 0.75rem; color: #475569;'>
        Multimodal Emotion Recognition • Speech + Text + Fusion
    </p>
</div>
""", unsafe_allow_html=True)
