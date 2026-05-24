"""
HTML templates for the Streamlit Multimodal Emotion Recognition app.
Extracted to a separate file to prevent linter false-positives from inline HTML/CSS.
"""

import numpy as np

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

HERO_SECTION = (
    "<div style='text-align: center; padding: 3.5rem 0 2rem 0; animation: fadeInDown 0.9s ease-out;'>"
    "<h1 style='font-size: 3.4rem; font-weight: 800; margin-bottom: 0.9rem; letter-spacing: -0.04em; color: #F8FAFC;'>"
    "🎙️ <span style='background: linear-gradient(135deg, #F7C948 0%, #E2B04E 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Multimodal Emotion Recognition</span>"
    "</h1>"
    "<p style='color: #CBD5E1; font-size: 1.1rem; font-weight: 600; margin-bottom: 2rem; letter-spacing: 0.08em; text-transform: uppercase;'>"
    "Speech · Text · Multimodal Fusion"
    "</p>"
    "<div style='margin-bottom: 1.8rem; display: flex; justify-content: center; gap: 1.25rem; flex-wrap: wrap;'>"
    "<span class='metric-badge' style='font-size: 0.95rem; padding: 0.65rem 1.6rem;'>Speech: 100.0%</span>"
    "<span class='metric-badge' style='font-size: 0.95rem; padding: 0.65rem 1.6rem;'>Text: 13.81%</span>"
    "<span class='metric-badge' style='font-size: 0.95rem; padding: 0.65rem 1.6rem; background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.35); color: #34D399;'>Fusion: 100.0%</span>"
    "</div>"
    "</div>"
)

SPEECH_INSIGHT = (
    "<div class='info-card-warning' style='box-shadow: 0 4px 20px rgba(245,158,11,0.1);'>"
    "<p style='font-weight: 800; margin-bottom: 0.5rem; color: #FBBF24; font-size: 1.15rem; letter-spacing: 0.03em;'>Speech Dominant</p>"
    "<p style='font-size: 0.95rem; color: #E2E8F0; margin: 0; line-height: 1.6; font-weight: 500;'>"
    "Acoustic features provide robust emotional cues through prosody and spectral patterns.</p>"
    "</div>"
)

TEXT_INSIGHT = (
    "<div class='info-card-warning' style='border-left-color: #EF4444; background: rgba(239, 68, 68, 0.08); box-shadow: 0 4px 20px rgba(239,68,68,0.1);'>"
    "<p style='font-weight: 800; margin-bottom: 0.5rem; color: #F87171; font-size: 1.15rem; letter-spacing: 0.03em;'>Text Limited</p>"
    "<p style='font-size: 0.95rem; color: #E2E8F0; margin: 0; line-height: 1.6; font-weight: 500;'>"
    "Isolated words lack emotional semantics. Performance improves significantly with full sentences.</p>"
    "</div>"
)

FUSION_INSIGHT = (
    "<div class='info-card' style='box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2); border-color: rgba(16, 185, 129, 0.4);'>"
    "<p style='font-weight: 800; margin-bottom: 0.5rem; color: #34D399; font-size: 1.15rem; letter-spacing: 0.03em;'>Fusion Integrates</p>"
    "<p style='font-size: 0.95rem; color: #E2E8F0; margin: 0; line-height: 1.6; font-weight: 500;'>"
    "Combines acoustic and semantic modalities for comprehensive, highly accurate emotion analysis.</p>"
    "</div>"
)

ARCHITECTURE_SECTION = (
    "<h3 style='font-size: 1.6rem; font-weight: 800; color: #FBBF24; margin-bottom: 1.5rem; border-bottom: 2px solid rgba(255,255,255,0.05); padding-bottom: 0.75rem;'>Pipeline Architecture Overview</h3>"
    ""
    "<p style='font-size: 1rem; font-weight: 800; color: #E2E8F0; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.15em;'>🎙️ Speech Emotion Pipeline</p>"
    "<div class='flow-container'>"
    "<span class='flow-step'>Audio Input</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>Trim Silence</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>Normalize</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>MFCC (40) + Δ + ΔΔ</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step accent-step' style='box-shadow: 0 0 15px rgba(16,185,129,0.3);'>CNN + BiLSTM + Attention</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>FC (128)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step accent-step' style='box-shadow: 0 0 15px rgba(16,185,129,0.3);'>Softmax (7 Class)</span>"
    "</div>"
    ""
    "<p style='font-size: 1rem; font-weight: 800; color: #E2E8F0; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.15em; margin-top: 2rem;'>📝 Text Emotion Pipeline</p>"
    "<div class='flow-container'>"
    "<span class='flow-step'>Text Transcript</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>Contextual Prompt</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step accent-step' style='box-shadow: 0 0 15px rgba(16,185,129,0.3);'>DistilBERT [CLS] Token</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>FC (256)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>FC (128)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step accent-step' style='box-shadow: 0 0 15px rgba(16,185,129,0.3);'>Softmax (7 Class)</span>"
    "</div>"
    ""
    "<p style='font-size: 1rem; font-weight: 800; color: #E2E8F0; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.15em; margin-top: 2rem;'>🔮 Multimodal Fusion Pipeline</p>"
    "<div class='flow-container'>"
    "<span class='flow-step'>Speech Encoder (128-dim)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>Text Encoder (256-dim)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step accent-step' style='box-shadow: 0 0 15px rgba(16,185,129,0.3);'>Multimodal Concat (384-dim)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>FC (256)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step'>FC (128)</span>"
    "<span class='flow-arrow'>➔</span>"
    "<span class='flow-step accent-step' style='box-shadow: 0 0 15px rgba(16,185,129,0.3);'>Softmax (7 Class)</span>"
    "</div>"
    ""
    "<h3 style='font-size: 1.6rem; font-weight: 800; color: #FBBF24; margin-top: 3rem; margin-bottom: 1.5rem; border-bottom: 2px solid rgba(255,255,255,0.05); padding-bottom: 0.75rem;'>Model Accuracy &amp; Performance</h3>"
    "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 1rem;'>"
    "<div style='background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 18px; padding: 2.5rem; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25); transition: transform 0.3s ease, box-shadow 0.3s ease;' onmouseover=\"this.style.transform='translateY(-4px)'; this.style.boxShadow='0 18px 40px rgba(0, 0, 0, 0.3)'\" onmouseout=\"this.style.transform='translateY(0)'; this.style.boxShadow='0 12px 35px rgba(0, 0, 0, 0.25)'\"> "
    "<p style='font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.12em; color: #94A3B8; margin: 0 0 0.75rem 0; font-weight: 700;'>Speech Performance</p>"
    "<h4 style='color: #FBBF24; margin: 0; font-size: 2.25rem; font-weight: 900;'>100.0% Accuracy</h4>"
    "<p style='font-size: 1rem; color: #CBD5E1; margin-top: 1rem; line-height: 1.7; font-weight: 500;'>Speech-only classification achieves perfect dataset-level accuracy in the reported model evaluation.</p>"
    "</div>"
    "<div style='background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 18px; padding: 2.5rem; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25); transition: transform 0.3s ease, box-shadow 0.3s ease;' onmouseover=\"this.style.transform='translateY(-4px)'; this.style.boxShadow='0 18px 40px rgba(0, 0, 0, 0.3)'\" onmouseout=\"this.style.transform='translateY(0)'; this.style.boxShadow='0 12px 35px rgba(0, 0, 0, 0.25)'\"> "
    "<p style='font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.12em; color: #94A3B8; margin: 0 0 0.75rem 0; font-weight: 700;'>Text Performance</p>"
    "<h4 style='color: #60A5FA; margin: 0; font-size: 2.25rem; font-weight: 900;'>13.81% Accuracy</h4>"
    "<p style='font-size: 1rem; color: #CBD5E1; margin-top: 1rem; line-height: 1.7; font-weight: 500;'>Text-only classification is weak on this dataset, reflecting limited semantic signal from isolated transcript input.</p>"
    "</div>"
    "<div style='background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 18px; padding: 2.5rem; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25); transition: transform 0.3s ease, box-shadow 0.3s ease;' onmouseover=\"this.style.transform='translateY(-4px)'; this.style.boxShadow='0 18px 40px rgba(0, 0, 0, 0.3)'\" onmouseout=\"this.style.transform='translateY(0)'; this.style.boxShadow='0 12px 35px rgba(0, 0, 0, 0.25)'\"> "
    "<p style='font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.12em; color: #94A3B8; margin: 0 0 0.75rem 0; font-weight: 700;'>Multimodal Fusion</p>"
    "<h4 style='color: #34D399; margin: 0; font-size: 2.25rem; font-weight: 900;'>100.0% Accuracy</h4>"
    "<p style='font-size: 1rem; color: #CBD5E1; margin-top: 1rem; line-height: 1.7; font-weight: 500;'>Fusion model performance mirrors speech-only classification, reflecting strong acoustic dominance in this evaluation.</p>"
    "</div>"
    "</div>"
)

FOOTER = (
    "<div class='footer'>"
    "<p class='footer-title'>Project Credits</p>"
    "<p class='footer-name'>Name: Pangoth Hemanth Nayak</p>"
    "<p class='footer-college'>Hyderabad Institute of Technology and Management (HITAM)</p>"
    "<hr class='footer-divider'>"
    "<p class='footer-internship'>IIITH RAP &nbsp;·&nbsp; Phase 2 Internship</p>"
    "<p class='footer-tagline'>Multimodal Emotion Recognition &nbsp;·&nbsp; Speech + Text + Fusion</p>"
    "</div>"
)

def top_prediction_html(probabilities) -> str:
    """Render a compact top-3 emotion prediction HTML block."""
    probs = np.asarray(probabilities).flatten()
    if probs.size != len(emotion_labels):
        return ""

    top_indices = np.argsort(probs)[::-1][:3]
    items = []
    for idx in top_indices:
        label = emotion_labels[idx].replace('_', ' ').title()
        score = probs[idx]
        items.append(
            f"<div class='top-3-item'><span class='top-3-label'>{label}</span>"
            f"<span class='top-3-value'>{score:.1%}</span></div>"
        )
    return (
        "<div class='top-3-list'>"
        "<p style='color: #94A3B8; font-size: 0.95rem; letter-spacing: 0.06em; margin-bottom: 0.75rem; text-transform: uppercase;'>Top 3 predictions</p>"
        + "".join(items) +
        "</div>"
    )


def prediction_card(
    emoji: str,
    emotion: str,
    confidence: float,
    model_name: str,
    is_fusion: bool = False,
    probabilities=None,
    extra_text: str = ""
) -> str:
    """Generate a prediction card HTML string."""
    border_style = (
        "border: 1px solid rgba(52, 211, 153, 0.25); box-shadow: 0 16px 45px rgba(16, 185, 129, 0.16);"
        if is_fusion else
        "border: 1px solid rgba(148, 163, 184, 0.18); box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18);"
    )
    conf_color = "#34D399" if is_fusion else "#FBBF24"
    card_background = "background: rgba(12, 18, 32, 0.9);"
    score_label = "Prediction Confidence"
    extra_html = (
        f"<p style='color: #94A3B8; font-size: 0.95rem; margin: 0.85rem 0 0 0; line-height: 1.5; font-weight: 600;'>{extra_text}</p>"
        if extra_text else ""
    )
    top3_html = top_prediction_html(probabilities) if probabilities is not None else ""

    return (
        f"<div class='prediction-card' style='{border_style} {card_background}'>"
        f"<p style='color: #94A3B8; font-size: 0.95rem; font-weight: 800; text-transform: uppercase; "
        f"letter-spacing: 0.16em; margin-bottom: 0.75rem;'>{model_name}</p>"
        f"<div style='font-size: 4.2rem; margin: 1.2rem 0; filter: drop-shadow(0 10px 20px rgba(0,0,0,0.25));'>{emoji}</div>"
        f"<h3 style='color: #F8FAFC; margin: 0.5rem 0; font-size: 1.75rem; font-weight: 900; letter-spacing: -0.03em;'>"
        f"{emotion.replace('_', ' ').title()}</h3>"
        f"<p style='color: {conf_color}; font-weight: 900; font-size: 2rem; margin: 0.25rem 0 0 0; line-height: 1.2;'>{confidence:.1%}</p>"
        f"<p style='color: #94A3B8; font-size: 0.95rem; margin: 0.35rem 0 0 0; line-height: 1.5; font-weight: 600;'>{score_label}</p>"
        f"{extra_html}"
        f"{top3_html}"
        "</div>"
    )

def confidence_label(emotion_name: str) -> str:
    """Generate a confidence distribution label HTML string."""
    return (
        f"<p style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC; "
        f"margin: 0.25rem 0; text-transform: capitalize; letter-spacing: 0.02em;'>{emotion_name.replace('_', ' ')}</p>"
    )
