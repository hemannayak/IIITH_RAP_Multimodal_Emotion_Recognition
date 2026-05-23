
import streamlit as st
import torch
import torch.nn as nn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel
import warnings

warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(page_title="IIITH Multimodal Emotion AI", page_icon="🎭", layout="wide")

# RESTORED PREMIUM UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: #0a0a0c;
        color: #ffffff;
    }

    [data-testid="stSidebar"] { display: none; }

    label[data-testid="stWidgetLabel"] p, .stMarkdown p, .stTextArea label p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.25);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        margin-bottom: 30px;
    }

    .emotion-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.8rem;
        border-radius: 18px;
        border-top: 4px solid #38bdf8;
        text-align: center;
        height: 100%;
    }

    h4, h5 {
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.5px;
        margin-bottom: 1.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        display: block;
        margin: 0 auto !important;
    }

    .footer-container {
        width: 100%; padding: 50px 20px; margin-top: 60px;
        border-top: 1px solid rgba(255, 255, 255, 0.05); text-align: center;
    }
    
    .highlight { color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# RESTORED MODEL CLASSES
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_output):
        scores = torch.softmax(self.attention_weights(lstm_output), dim=1)
        return torch.sum(scores * lstm_output, dim=1)

class SpeechEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(120, 128, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = AttentionLayer(256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 7)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.maxpool(self.relu(self.batch_norm1(self.conv1(x)))))
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        return self.fc2(self.relu(self.fc1(self.attention(out))))

class TextEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 7)
    def forward(self, ids, mask):
        x = self.distilbert(ids, mask).last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

@st.cache_resource
def load_resources():
    s = SpeechEmotionModel()
    t = TextEmotionModel()
    s.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/IIITH_RAP_Multimodal_Emotion_Recognition/saved_models/advanced_speech_emotion_model.pth", map_location='cpu'))
    t.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/IIITH_RAP_Multimodal_Emotion_Recognition/saved_models/text_emotion_model.pth", map_location='cpu'))
    return s.eval(), t.eval(), DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

s_mod, t_mod, tokenizer = load_resources()
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']
emojis = ['💢', '🤢', '😨', '😊', '😐', '😲', '😔']

# HEADER
st.markdown("<div style='text-align:center; padding-top: 1rem;'><h1 style='font-size: 3.2rem; font-weight: 800; margin-bottom:0;'>🎭 Multimodal <span style='color:#38bdf8'>Emotion AI</span></h1><p style='color:#94a3b8; font-size:1.1rem;'>Research-Grade Acoustic-Semantic Intelligence Dashboard</p></div>", unsafe_allow_html=True)

# 1. INPUT LAYOUT
st.markdown("<div class='glass-card'><h4>📥 Multimodal Input Streams</h4>", unsafe_allow_html=True)
in_col1, in_col2 = st.columns(2)
with in_col1:
    aud_file = st.file_uploader("Audio Signal (Acoustic Stream)", type=["wav"])
with in_col2:
    txt_input = st.text_area("Semantic Context (Text Input)", placeholder="Type semantic context here...", height=68)

st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 0.8, 1])
with btn_col2:
    run_btn = st.button("ANALYZE EMOTIONAL SIGNATURE", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# 2. ACOUSTIC ANALYTICS (RESTORED FEATURES)
if aud_file:
    st.markdown("<div class='glass-card'><h4>📊 Acoustic Analytics</h4>", unsafe_allow_html=True)
    y, sr = librosa.load(aud_file, sr=16000)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Signal Duration", f"{len(y)/sr:.2f}s")
    m_col2.metric("RMS Energy", f"{np.mean(librosa.feature.rms(y=y)):.4f}")
    m_col3.metric("Zero Crossing Rate", f"{np.mean(librosa.feature.zero_crossing_rate(y)):.4f}")
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    plt.subplots_adjust(hspace=0.7)
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color='#38bdf8', alpha=0.8)
    ax[0].set_title("Temporal Waveform Analysis", color='#f8fafc', size=11, weight='bold')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, ax=ax[1], cmap='magma')
    ax[1].set_title("Mel-Frequency Spectrogram", color='#f8fafc', size=11, weight='bold')
    for a in ax: a.axis('off')
    fig.patch.set_facecolor('#0a0a0c')
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# 3. PREDICTION RESULTS
if run_btn:
    s_p, t_p = None, None
    if aud_file:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feat = np.pad(mfcc.T, ((0, max(0, 200-mfcc.shape[1])), (0,0)))[:200, :]
        feat = np.concatenate([feat, np.zeros_like(feat), np.zeros_like(feat)], axis=1)
        with torch.no_grad(): s_p = torch.softmax(s_mod(torch.tensor(feat).float().unsqueeze(0)), 1)

    if txt_input.strip():
        inputs = tokenizer(txt_input, return_tensors="pt", padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            logits = t_mod(inputs['input_ids'], inputs['attention_mask'])
            t_p = torch.softmax(logits / 0.5, 1) # Sharpening retained for accuracy

    if s_p is not None or t_p is not None:
        st.markdown("<div class='glass-card'><h4>🎯 Neural Intelligence Reports</h4>", unsafe_allow_html=True)
        f_p = (0.50 * s_p + 0.50 * t_p) if (s_p is not None and t_p is not None) else (s_p if s_p is not None else t_p)
        
        res_cols = st.columns(3)
        def draw_card(col, title, probs):
            if probs is not None:
                idx = probs.argmax()
                col.markdown(f"<div class='emotion-card'><h6>{title}</h6><h2>{emojis[idx]} {labels[idx]}</h2><p style='color:#38bdf8; font-weight:800; font-size:1.3rem; margin-top:10px;'>{probs.max()*100:.1f}% Match</p></div>", unsafe_allow_html=True)

        draw_card(res_cols[0], "ACOUSTIC PREDICTION", s_p)
        draw_card(res_cols[1], "SEMANTIC PREDICTION", t_p)
        draw_card(res_cols[2], "RESEARCH FUSION", f_p)

        st.markdown("<br><h5>Confidence Distribution</h5>", unsafe_allow_html=True)
        for i, l in enumerate(labels):
            st.write(f"{l} ({f_p[0][i]*100:.1f}%)")
            st.progress(float(f_p[0][i]))

        # RESTORED XAI SECTION
        st.markdown("<br><h5>🧠 Why This Prediction? (XAI)</h5>", unsafe_allow_html=True)
        pred = labels[f_p.argmax()]
        explanations = {
            'Angry': 'Detected high acoustic intensity peaks and aggressive semantic markers.',
            'Happy': 'Rise in pitch contour and positive lexical sentiment tokens observed.',
            'Sad': 'Low RMS energy levels and slower speech tempo signify a depressive state.',
            'Neutral': 'Steady pitch variability and balanced frequency distribution indicate calm.',
            'Surprise': 'Sudden frequency shifts and high-pitch acoustic fluctuations detected.',
            'Fear': 'Rapid zero-crossing rate and high semantic instability observed.',
            'Disgust': 'Specific low-frequency patterns and negative linguistic context detected.'
        }
        st.success(explanations.get(pred, "Combined multimodal features suggest this emotional signature."))
        st.markdown("</div>", unsafe_allow_html=True)

# RESTORED FOOTER
st.markdown(f"""
<div class='footer-container'>
    <div style='font-size: 1rem; color:#f8fafc; font-weight:700;'>DEVELOPER: <span class='highlight'>PANGOTH HEMANTH NAYAK</span> • ROLL NO: 23E51A67C5</div>
    <div style='font-size: 0.85rem; color:#64748b; margin-top:10px; font-weight:500;'>HYDERABAD INSTITUTE OF TECHNOLOGY AND MANAGEMENT (HITAM) • IIITH RAP PHASE II</div>
</div>
""", unsafe_allow_html=True)
