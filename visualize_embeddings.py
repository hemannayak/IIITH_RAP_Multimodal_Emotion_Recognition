"""
Embedding Visualization Script
Generates t-SNE and PCA plots for emotion cluster separability analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch

from inference.speech_inference import load_speech_model
from inference.text_inference import load_text_model
from inference.fusion_inference import load_fusion_model, extract_speech_features
from transformers import DistilBertTokenizer

# Emotion labels and colors
EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", 
    "neutral", "pleasant_surprise", "sad"
]

EMOTION_COLORS = {
    "angry": "#e74c3c",
    "disgust": "#9b59b6",
    "fear": "#3498db",
    "happy": "#f39c12",
    "neutral": "#95a5a6",
    "pleasant_surprise": "#1abc9c",
    "sad": "#34495e"
}

# Model paths
SPEECH_MODEL_PATH = "saved_models/advanced_speech_emotion_model.pth"
TEXT_MODEL_PATH = "saved_models/text_emotion_model.pth"
FUSION_MODEL_PATH = "saved_models/multimodal_fusion_model.pth"

# Results directory
VIZ_DIR = "Results/visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)


def create_metadata_from_dataset():
    """Create metadata from TESS dataset"""
    print("Creating metadata from dataset...")
    
    dataset_path = "dataset/TESS Toronto emotional speech set data"
    data = []
    
    emotion_folders = {
        "OAF_angry": "angry",
        "OAF_disgust": "disgust",
        "OAF_Fear": "fear",
        "OAF_happy": "happy",
        "OAF_neutral": "neutral",
        "OAF_Pleasant_surprise": "pleasant_surprise",
        "OAF_Sad": "sad",
        "YAF_angry": "angry",
        "YAF_disgust": "disgust",
        "YAF_fear": "fear",
        "YAF_happy": "happy",
        "YAF_neutral": "neutral",
        "YAF_pleasant_surprised": "pleasant_surprise",
        "YAF_sad": "sad"
    }
    
    for folder, emotion in emotion_folders.items():
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    parts = filename.replace('.wav', '').split('_')
                    if len(parts) >= 2:
                        word = parts[1]
                        data.append({
                            'audio_path': os.path.join(folder_path, filename),
                            'transcript': word,
                            'emotion': emotion,
                            'label': EMOTION_LABELS.index(emotion)
                        })
    
    df = pd.DataFrame(data)
    print(f"Created metadata with {len(df)} samples")
    return df


def extract_speech_embeddings(test_df, model):
    """Extract speech embeddings from fc1 layer"""
    print("\nExtracting speech embeddings...")
    
    embeddings = []
    labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    for idx, row in test_df.iterrows():
        try:
            # Extract features
            features = extract_speech_features(row['audio_path'])
            speech_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Forward pass through model layers
            with torch.no_grad():
                # Permute for Conv1d
                x = speech_tensor.permute(0, 2, 1)
                
                # Conv + BatchNorm + ReLU + MaxPool + Dropout
                x = model.conv1(x)
                x = model.batch_norm1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.dropout(x)
                
                # Permute back for LSTM
                x = x.permute(0, 2, 1)
                
                # BiLSTM
                lstm_out, _ = model.bilstm(x)
                
                # Attention
                attention_out = model.attention(lstm_out)
                
                # FC1 layer (this is our embedding)
                embedding = model.fc1(attention_out)
                
                embeddings.append(embedding.cpu().numpy().flatten())
                labels.append(row['emotion'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
                
        except Exception as e:
            print(f"Error processing {row['audio_path']}: {e}")
            continue
    
    return np.array(embeddings), labels


def extract_text_embeddings(test_df, model):
    """Extract text embeddings from DistilBERT"""
    print("\nExtracting text embeddings...")
    
    embeddings = []
    labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()
    
    for idx, row in test_df.iterrows():
        try:
            # Contextual prompting (matching text-only training)
            transcript = f"The speaker emotionally expressed the word {row['transcript']}"
            
            encoding = tokenizer(
                transcript,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embedding = model.fc1(cls_embedding)
                embedding = model.relu(embedding)
                
                embeddings.append(embedding.cpu().numpy().flatten())
                labels.append(row['emotion'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
                
        except Exception as e:
            print(f"Error processing {row['transcript']}: {e}")
            continue
    
    return np.array(embeddings), labels


def extract_fusion_embeddings(test_df, model):
    """Extract fusion embeddings"""
    print("\nExtracting fusion embeddings...")
    
    embeddings = []
    labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()
    
    for idx, row in test_df.iterrows():
        try:
            # Speech features
            speech_features = extract_speech_features(row['audio_path'])
            speech_tensor = torch.tensor(speech_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Text features (raw word for fusion)
            encoding = tokenizer(
                row['transcript'],
                padding="max_length",
                truncation=True,
                max_length=16,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            with torch.no_grad():
                _, fusion_embedding = model(speech_tensor, input_ids, attention_mask)
                
                embeddings.append(fusion_embedding.cpu().numpy().flatten())
                labels.append(row['emotion'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
                
        except Exception as e:
            print(f"Error processing {row['audio_path']}: {e}")
            continue
    
    return np.array(embeddings), labels


def plot_tsne(embeddings, labels, title, filename):
    """Generate t-SNE visualization"""
    print(f"\nGenerating t-SNE for {title}...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    for emotion in EMOTION_LABELS:
        mask = np.array(labels) == emotion
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion.replace('_', ' ').title(),
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")


def plot_pca(embeddings, labels, title, filename):
    """Generate PCA visualization"""
    print(f"\nGenerating PCA for {title}...")
    
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    explained_var = pca.explained_variance_ratio_
    
    plt.figure(figsize=(12, 10))
    
    for emotion in EMOTION_LABELS:
        mask = np.array(labels) == emotion
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion.replace('_', ' ').title(),
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.title(
        f"{title}\n(PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%} variance explained)",
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1%})', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1%})', fontsize=14)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")


def main():
    """Main visualization pipeline"""
    print("\n" + "="*60)
    print("EMOTION EMBEDDING VISUALIZATION")
    print("="*60 + "\n")
    
    # Step 1: Create metadata and test split
    df = create_metadata_from_dataset()
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=42
    )
    
    print(f"\nTest set size: {len(test_df)} samples")
    
    # Step 2: Load models
    print("\nLoading models...")
    speech_model = load_speech_model(SPEECH_MODEL_PATH)
    text_model = load_text_model(TEXT_MODEL_PATH)
    fusion_model = load_fusion_model(FUSION_MODEL_PATH)
    print("Models loaded successfully!")
    
    # Step 3: Extract embeddings
    speech_embeddings, speech_labels = extract_speech_embeddings(test_df, speech_model)
    text_embeddings, text_labels = extract_text_embeddings(test_df, text_model)
    fusion_embeddings, fusion_labels = extract_fusion_embeddings(test_df, fusion_model)
    
    # Step 4: Generate t-SNE visualizations
    print("\n" + "="*60)
    print("GENERATING T-SNE VISUALIZATIONS")
    print("="*60)
    
    plot_tsne(
        speech_embeddings, speech_labels,
        "Speech Emotion Embeddings (t-SNE)",
        "speech_tsne.png"
    )
    
    plot_tsne(
        text_embeddings, text_labels,
        "Text Emotion Embeddings (t-SNE)",
        "text_tsne.png"
    )
    
    plot_tsne(
        fusion_embeddings, fusion_labels,
        "Fusion Emotion Embeddings (t-SNE)",
        "fusion_tsne.png"
    )
    
    # Step 5: Generate PCA visualizations
    print("\n" + "="*60)
    print("GENERATING PCA VISUALIZATIONS")
    print("="*60)
    
    plot_pca(
        speech_embeddings, speech_labels,
        "Speech Emotion Embeddings (PCA)",
        "speech_pca.png"
    )
    
    plot_pca(
        text_embeddings, text_labels,
        "Text Emotion Embeddings (PCA)",
        "text_pca.png"
    )
    
    plot_pca(
        fusion_embeddings, fusion_labels,
        "Fusion Emotion Embeddings (PCA)",
        "fusion_pca.png"
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nVisualizations saved to: {VIZ_DIR}/")
    print("\nGenerated files:")
    print("  - speech_tsne.png")
    print("  - text_tsne.png")
    print("  - fusion_tsne.png")
    print("  - speech_pca.png")
    print("  - text_pca.png")
    print("  - fusion_pca.png")


if __name__ == "__main__":
    main()
