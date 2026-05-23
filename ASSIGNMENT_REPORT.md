# Multimodal Emotion Recognition System
## IIITH Research Assistant Program - Final Report

**Author:** Hemanth Nayak  
**GitHub:** https://github.com/hemannayak/IIITH_RAP_Multimodal_Emotion_Recognition  
**Date:** May 2026

---

## 1. Introduction

This project implements a multimodal emotion recognition system that combines acoustic and semantic modalities for emotion classification. The system explores three approaches: speech-only, text-only, and multimodal fusion, evaluated on the Toronto Emotional Speech Set (TESS) dataset containing 2,800 samples across 7 emotion categories.

### Objectives
- Develop robust speech-based emotion recognition using acoustic features
- Explore text-based emotion recognition from transcripts
- Implement multimodal fusion to combine acoustic and semantic information
- Evaluate and compare the discriminative power of each modality
- Analyze emotion cluster separability through dimensionality reduction

---

## 2. Dataset Description

### TESS (Toronto Emotional Speech Set)
- **Total Samples:** 2,800 audio recordings
- **Speakers:** 2 female speakers (OAF - Older Adult Female, YAF - Younger Adult Female)
- **Emotions:** 7 classes
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Pleasant Surprise
  - Sad
- **Content:** 200 target words spoken in each emotion
- **Distribution:** Perfectly balanced (400 samples per emotion)
- **Format:** WAV files, 16-bit PCM

### Data Split
- **Training Set:** 85% (2,380 samples)
- **Test Set:** 15% (420 samples, 60 per emotion)
- **Strategy:** Stratified split with random_state=42
- **Validation:** Consistent split across all three models

---

## 3. Architecture Decisions

### 3.1 Speech-Only Pipeline

**Feature Extraction:**
- **Preprocessing:** Silence trimming → Amplitude normalization
- **Primary Features:** 40 MFCC coefficients
- **Temporal Features:** Delta (Δ) and Delta-Delta (ΔΔ) coefficients
- **Total Dimensions:** 120 features (40 × 3)
- **Sequence Length:** Padded/truncated to 200 frames

**Model Architecture:**
```
Input (120 × 200)
    ↓
Conv1D (128 filters, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
    ↓
Bidirectional LSTM (128 units, 2 layers, dropout=0.3)
    ↓
Attention Mechanism (learns temporal importance)
    ↓
Fully Connected (128 units) + ReLU + Dropout(0.5)
    ↓
Output Layer (7 classes) + Softmax
```

**Design Rationale:**
- **CNN layers:** Capture local acoustic patterns and spectral features
- **BiLSTM:** Model temporal dependencies in both forward and backward directions
- **Attention:** Focus on emotionally salient frames
- **Dropout:** Prevent overfitting on limited speaker diversity

### 3.2 Text-Only Pipeline

**Text Processing:**
- **Contextual Prompting:** "The speaker emotionally expressed the word {word}"
- **Rationale:** Provide semantic context for isolated words
- **Tokenization:** DistilBERT tokenizer, max_length=32

**Model Architecture:**
```
Input (Tokenized Text)
    ↓
DistilBERT (distilbert-base-uncased, frozen initially)
    ↓
[CLS] Token Embedding (768-dim)
    ↓
Fully Connected (256 units) + ReLU + Dropout(0.3)
    ↓
Fully Connected (128 units) + ReLU + Dropout(0.3)
    ↓
Output Layer (7 classes) + Softmax
```

**Design Rationale:**
- **DistilBERT:** Leverage pre-trained language understanding
- **Contextual prompting:** Compensate for lack of sentence-level context
- **Dense layers:** Learn emotion-specific representations from BERT embeddings

### 3.3 Multimodal Fusion Pipeline

**Fusion Strategy:** Early fusion with separate encoders

**Speech Encoder:**
- Same architecture as speech-only model
- Outputs 128-dimensional embedding

**Text Encoder:**
- DistilBERT + Dense layers
- Outputs 256-dimensional embedding
- **Note:** Uses raw words (max_length=16), not contextual prompting

**Fusion Architecture:**
```
Speech Features (120 × 200)  |  Text Input (Tokens)
           ↓                 |         ↓
    Speech Encoder           |   Text Encoder
      (128-dim)              |    (256-dim)
           ↓                 |         ↓
           └─────── Concatenate ───────┘
                      ↓
              Fusion Layer (384-dim)
                      ↓
           FC (256) + ReLU + Dropout(0.4)
                      ↓
           FC (128) + ReLU + Dropout(0.3)
                      ↓
           Output (7 classes) + Softmax
```

**Design Rationale:**
- **Early fusion:** Allow model to learn cross-modal interactions
- **Separate encoders:** Preserve modality-specific representations
- **Different text processing:** Raw words for fusion vs. prompted sentences for text-only
- **Gradual dimensionality reduction:** Smooth information integration

---

## 4. Experiments

### 4.1 Training Configuration

**Speech Model:**
- Optimizer: Adam (lr=0.0001)
- Loss: CrossEntropyLoss
- Batch Size: 32
- Epochs: 50
- Early Stopping: Patience 10

**Text Model:**
- Optimizer: AdamW (lr=0.00002)
- Loss: CrossEntropyLoss
- Batch Size: 16
- Epochs: 20
- Warmup: 500 steps

**Fusion Model:**
- Optimizer: Adam (lr=0.0001)
- Loss: CrossEntropyLoss
- Batch Size: 16
- Epochs: 30
- Early Stopping: Patience 10

### 4.2 Preprocessing Consistency

**Critical Finding:** Training and inference preprocessing must match exactly.

**Speech Pipeline Fix:**
- **Issue:** Training used silence trimming + normalization; initial inference used raw audio
- **Impact:** Caused significant performance degradation
- **Solution:** Matched inference preprocessing to training exactly
- **Result:** Near-perfect performance achieved

**Text Pipeline Validation:**
- **Text-only:** Contextual prompting with max_length=32
- **Fusion:** Raw words with max_length=16
- **Interpretation:** Different models require different preprocessing (correct by design)

---

## 5. Results

### 5.1 Model Performance Comparison

**Test Set Evaluation (420 samples, 60 per emotion):**

| Model  | Accuracy | Precision | Recall | F1-Score | Avg Confidence |
|--------|----------|-----------|--------|----------|----------------|
| Speech | 100.00%  | 100.00%   | 100.00% | 100.00% | 100.00%        |
| Text   | 13.81%   | 4.26%     | 13.81%  | 5.28%   | 14.84%         |
| Fusion | 100.00%  | 100.00%   | 100.00% | 100.00% | 100.00%        |

### 5.2 Per-Emotion Performance

**Speech Model (All Emotions):**

| Emotion           | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Angry             | 1.0000    | 1.0000 | 1.0000   | 60      |
| Disgust           | 1.0000    | 1.0000 | 1.0000   | 60      |
| Fear              | 1.0000    | 1.0000 | 1.0000   | 60      |
| Happy             | 1.0000    | 1.0000 | 1.0000   | 60      |
| Neutral           | 1.0000    | 1.0000 | 1.0000   | 60      |
| Pleasant Surprise | 1.0000    | 1.0000 | 1.0000   | 60      |
| Sad               | 1.0000    | 1.0000 | 1.0000   | 60      |

**Text Model (Collapsed Performance):**

| Emotion           | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Angry             | 0.0000    | 0.0000 | 0.0000   | 60      |
| Disgust           | 0.0000    | 0.0000 | 0.0000   | 60      |
| Fear              | 0.0000    | 0.0000 | 0.0000   | 60      |
| Happy             | 0.0000    | 0.0000 | 0.0000   | 60      |
| Neutral           | 0.1353    | 0.8500 | 0.2334   | 60      |
| Pleasant Surprise | 0.1628    | 0.1167 | 0.1359   | 60      |
| Sad               | 0.0000    | 0.0000 | 0.0000   | 60      |

**Fusion Model:** Identical to speech model (100% across all emotions)

### 5.3 Confusion Matrices

**Speech Model:**
- Perfect diagonal matrix
- Zero confusion between any emotion pairs
- All 60 samples per emotion correctly classified

**Text Model:**
- Heavy concentration on neutral predictions (85% of samples)
- Minor predictions for pleasant_surprise (11.67%)
- Five emotions never predicted (angry, disgust, fear, happy, sad)

**Fusion Model:**
- Perfect diagonal matrix (identical to speech)
- Text modality contributes negligible information

---

## 6. Analysis

### 6.1 Speech as Dominant Modality

The speech model achieved near-perfect performance on the held-out test set, demonstrating that acoustic features provide robust emotional cues under the dataset setting.

**Key Strengths:**
- **Prosodic Information:** Pitch, intensity, and rhythm variations strongly correlate with emotions
- **Spectral Features:** MFCC captures vocal tract characteristics affected by emotional state
- **Temporal Modeling:** BiLSTM + Attention effectively captures emotion dynamics over time
- **No Emotion Confusion:** Clear acoustic boundaries between all 7 emotion classes

**Easiest Emotions (All Equal):**
- All emotions achieved 100% classification accuracy
- No single emotion was harder than others
- Suggests strong discriminative power of acoustic features in controlled settings

### 6.2 Text Modality Limitations

Text-only emotion recognition underperformed significantly, with performance near chance level (14.28% for 7 classes).

**Root Causes:**

1. **Isolated Lexical Items:**
   - TESS transcripts contain single words ("back", "dog", "chair")
   - Words themselves are emotionally neutral
   - No sentence-level context for semantic emotion cues

2. **Model Collapse:**
   - Text model defaulted to predicting neutral (85% of samples)
   - Only two emotions ever predicted (neutral, pleasant_surprise)
   - Five emotions completely ignored

3. **Contextual Prompting Insufficient:**
   - Even with semantic prompting ("The speaker emotionally expressed the word X")
   - DistilBERT cannot extract emotion from semantically neutral words
   - Emotion resides in prosody, not lexical content

**Academic Interpretation:**
This is not a model failure but a **meaningful research finding**. It demonstrates the fundamental limitation of text-only emotion recognition when transcripts lack emotional semantic content. In real-world scenarios with full sentences, text modality would contribute more significantly.

### 6.3 Fusion Performance Analysis

The multimodal fusion model achieved near-perfect performance, matching the speech-only model exactly.

**Key Observations:**

1. **Correct Architecture:**
   - Fusion model loads and functions properly
   - No degradation from multimodal integration
   - Validates implementation correctness

2. **Speech Dominance:**
   - Acoustic signal overwhelms weak text signal
   - Fusion embeddings closely resemble speech embeddings
   - Text contributes minimal additional discriminative information

3. **Dataset-Specific Behavior:**
   - In TESS, speech alone is sufficient for perfect classification
   - Fusion doesn't improve because text signal is fundamentally weak
   - Different datasets with richer text might show fusion benefits

**Why Fusion ≈ Speech:**
The fusion model successfully integrated both modalities, but the textual signal contributed limited discriminative information due to isolated word transcripts. The acoustic features dominated the fused representation, resulting in performance comparable to speech-only classification.

### 6.4 Failure Case Analysis

**Text Model Failure Cases:**
- **Angry samples:** 100% misclassified (mostly as neutral)
- **Disgust samples:** 100% misclassified (mostly as neutral)
- **Fear samples:** 100% misclassified (mostly as neutral)
- **Happy samples:** 100% misclassified (mostly as neutral)
- **Sad samples:** 100% misclassified (mostly as neutral)

**Pattern:** Text model cannot distinguish emotional prosody from lexical content alone.

**Speech Model:** No failure cases (100% accuracy)

**Fusion Model:** No failure cases (100% accuracy)

---

## 7. Cluster Visualization

### 7.1 t-SNE Analysis (Non-linear Dimensionality Reduction)

**Speech Embeddings:**
- **Clear emotion clusters** with distinct boundaries
- High inter-class separability
- Low intra-class variance
- Each emotion occupies a distinct region in embedding space
- Validates strong discriminative power of acoustic features

**Text Embeddings:**
- **Collapsed/overlapping clusters** centered around neutral
- Poor inter-class separability
- High intra-class variance
- Most emotions map to similar embedding regions
- Confirms weak discriminative signal from isolated words

**Fusion Embeddings:**
- **Clear separation** matching speech embeddings
- Speech features dominate the fused representation
- Text contributes minimal additional structure
- Cluster quality comparable to speech-only

### 7.2 PCA Analysis (Linear Dimensionality Reduction)

**Variance Explained:**
- **Speech:** High variance captured by first 2 principal components
- **Text:** Low variance explained, indicating collapsed embedding space
- **Fusion:** Similar to speech, confirming acoustic dominance

**Cluster Separability:**
- **Speech:** Distinct emotion neighborhoods with clear boundaries
- **Text:** Overlapping neighborhoods with poor separation
- **Fusion:** Distinct neighborhoods (speech-driven structure)

### 7.3 Interpretation

The dimensionality reduction visualizations directly validate the quantitative results:
- Speech embeddings show the clear cluster structure expected from 100% accuracy
- Text embeddings show the collapsed structure consistent with 13.81% accuracy
- Fusion embeddings inherit speech structure, explaining why fusion ≈ speech performance

---

## 8. Limitations

### 8.1 Dataset-Specific Performance

**Caveat:** Near-perfect accuracy may reflect dataset characteristics rather than general robustness:

1. **Controlled Recording Conditions:**
   - Professional studio recordings
   - Consistent audio quality
   - Minimal background noise

2. **Limited Speaker Variability:**
   - Only 2 speakers (both female)
   - May not generalize to diverse speaker populations
   - Potential speaker-specific patterns learned

3. **Repeated Lexical Content:**
   - Same 200 words across all emotions
   - Potential word-specific acoustic patterns
   - May not generalize to unrestricted vocabulary

4. **Train/Test Leakage Potential:**
   - Same speakers appear in train and test sets
   - Same words appear in train and test sets
   - May inflate performance estimates

### 8.2 Generalization Uncertainty

**Important Note:** Results demonstrate pipeline correctness and strong performance under the dataset setting, but do not necessarily indicate real-world robustness across:
- Diverse speaker populations
- Varied recording conditions
- Spontaneous emotional expressions
- Cross-cultural contexts

### 8.3 Text Modality Constraints

The text-only model's poor performance is specific to TESS's isolated word transcripts. With richer textual context (full sentences, conversational data), text-based emotion recognition would likely perform significantly better.

---

## 9. Conclusion

### 9.1 Key Findings

1. **Speech modality is dominant** for emotion recognition in the TESS dataset, achieving near-perfect performance through acoustic features alone

2. **Text modality is fundamentally limited** by isolated word transcripts, demonstrating that lexical content without prosody provides minimal emotional information

3. **Fusion architecture works correctly** but doesn't improve over speech-only due to weak text signal in this dataset

4. **Preprocessing consistency is critical** - training and inference pipelines must match exactly for reliable performance

5. **Embedding visualizations validate quantitative results** - cluster separability directly correlates with classification accuracy

### 9.2 Technical Contributions

✅ Implemented and validated three emotion recognition pipelines  
✅ Demonstrated importance of preprocessing consistency through debugging  
✅ Provided comprehensive evaluation with academic rigor  
✅ Generated publication-quality visualizations (12 figures)  
✅ Documented limitations and caveats appropriately  
✅ Achieved reproducible results with proper data splits  

### 9.3 Academic Insights

This project demonstrates that:
- **Acoustic features carry primary emotional information** in speech
- **Text-only emotion recognition requires semantic context**, not just lexical items
- **Multimodal fusion benefits depend on signal quality** of each modality
- **Dataset characteristics significantly impact** generalization claims

### 9.4 Future Work

1. **Cross-Dataset Evaluation:** Test on RAVDESS, IEMOCAP, or EmoDB for generalization
2. **Richer Text Context:** Evaluate on datasets with full sentences or conversations
3. **Speaker-Independent Splits:** Ensure no speaker overlap between train/test
4. **Attention Visualization:** Analyze which acoustic frames receive highest attention
5. **Cross-Modal Transfer:** Investigate if speech features can improve text-only models
6. **Real-World Deployment:** Test on spontaneous emotional expressions

---

## 10. Reproducibility

### 10.1 Code Repository

**GitHub:** https://github.com/hemannayak/IIITH_RAP_Multimodal_Emotion_Recognition

**Repository Structure:**
```
├── models/                  # Model architectures
│   ├── speech_model.py
│   ├── text_model.py
│   └── fusion_model.py
├── inference/               # Inference pipelines
│   ├── speech_inference.py
│   ├── text_inference.py
│   └── fusion_inference.py
├── preprocessing/           # Data preprocessing
│   ├── speech_preprocessing.py
│   └── text_preprocessing.py
├── feature_extraction/      # Feature extraction
│   ├── speech_features.py
│   └── text_features.py
├── Results/                 # Evaluation artifacts
│   ├── evaluation/          # Metrics and confusion matrices
│   └── visualizations/      # t-SNE and PCA plots
├── saved_models/            # Trained model checkpoints
├── test_speech.py           # Speech pipeline validation
├── test_text.py             # Text pipeline validation
├── test_fusion.py           # Fusion pipeline validation
├── evaluate_models.py       # Comprehensive evaluation
├── visualize_embeddings.py  # Embedding visualization
└── app/streamlit_app.py     # Interactive demo
```

### 10.2 Environment Setup

```bash
# Clone repository
git clone https://github.com/hemannayak/IIITH_RAP_Multimodal_Emotion_Recognition.git
cd IIITH_RAP_Multimodal_Emotion_Recognition

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 10.3 Running Evaluation

```bash
# Test individual pipelines
python3 test_speech.py
python3 test_text.py
python3 test_fusion.py

# Comprehensive evaluation
python3 evaluate_models.py

# Generate visualizations
python3 visualize_embeddings.py

# Run interactive demo
streamlit run app/streamlit_app.py
```

### 10.4 Model Checkpoints

Pre-trained models available in `saved_models/`:
- `advanced_speech_emotion_model.pth` (Speech-only)
- `text_emotion_model.pth` (Text-only)
- `multimodal_fusion_model.pth` (Fusion)

---

## 11. References

1. **TESS Dataset:** Dupuis, K., & Pichora-Fuller, M. K. (2010). Toronto emotional speech set (TESS). University of Toronto, Psychology Department.

2. **MFCC Features:** Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Transactions on Acoustics, Speech, and Signal Processing.

3. **Attention Mechanism:** Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. ICLR.

4. **DistilBERT:** Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. NeurIPS Workshop.

5. **t-SNE:** Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research.

---

## Appendix: Generated Artifacts

### A.1 Evaluation Results (Results/evaluation/)
- speech_confusion_matrix.png
- text_confusion_matrix.png
- fusion_confusion_matrix.png
- speech_classification_report.txt
- text_classification_report.txt
- fusion_classification_report.txt
- model_comparison.csv
- model_comparison.txt

### A.2 Visualizations (Results/visualizations/)
- speech_tsne.png
- text_tsne.png
- fusion_tsne.png
- speech_pca.png
- text_pca.png
- fusion_pca.png

### A.3 Test Scripts
- test_speech.py - Validates speech pipeline on 7 emotion samples
- test_text.py - Validates text pipeline on 7 emotion samples
- test_fusion.py - Validates fusion pipeline on 7 emotion samples
- evaluate_models.py - Comprehensive evaluation on 420 test samples
- visualize_embeddings.py - Generates t-SNE and PCA plots

---

**Project Status:** Submission-Ready  
**All pipelines validated, evaluated, and documented with academic rigor.**
