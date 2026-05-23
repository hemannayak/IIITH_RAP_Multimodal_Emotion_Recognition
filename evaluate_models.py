"""
Comprehensive Model Evaluation Script
Generates confusion matrices, classification reports, and comparison visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import torch

from inference.speech_inference import load_speech_model, predict_speech_emotion
from inference.text_inference import load_text_model, predict_text_emotion
from inference.fusion_inference import load_fusion_model, predict_fusion_emotion


# Emotion labels
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

# Model paths
SPEECH_MODEL_PATH = "saved_models/advanced_speech_emotion_model.pth"
TEXT_MODEL_PATH = "saved_models/text_emotion_model.pth"
FUSION_MODEL_PATH = "saved_models/multimodal_fusion_model.pth"

# Results directory
RESULTS_DIR = "Results/evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_metadata_from_dataset():
    """
    Create metadata from TESS dataset structure
    """
    print("Creating metadata from dataset...")
    
    dataset_path = "dataset/TESS Toronto emotional speech set data"
    
    data = []
    
    # Emotion folder mapping
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
                    # Extract word from filename
                    # Format: OAF_word_emotion.wav or YAF_word_emotion.wav
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
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    return df


def create_test_split(df):
    """
    Create test split matching training configuration
    test_size=0.15, random_state=42, stratified
    """
    print("\nCreating test split...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=42
    )
    
    print(f"Test set size: {len(test_df)} samples")
    print(f"Test emotion distribution:\n{test_df['emotion'].value_counts()}")
    
    return test_df


def evaluate_speech_model(test_df, model):
    """
    Evaluate speech-only model
    """
    print("\n" + "="*60)
    print("EVALUATING SPEECH MODEL")
    print("="*60)
    
    y_true = []
    y_pred = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        try:
            result = predict_speech_emotion(row['audio_path'], model)
            
            y_true.append(row['emotion'])
            y_pred.append(result['emotion'])
            confidences.append(result['confidence'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
                
        except Exception as e:
            print(f"Error processing {row['audio_path']}: {e}")
            continue
    
    return y_true, y_pred, confidences


def evaluate_text_model(test_df, model):
    """
    Evaluate text-only model
    """
    print("\n" + "="*60)
    print("EVALUATING TEXT MODEL")
    print("="*60)
    
    y_true = []
    y_pred = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        try:
            result = predict_text_emotion(row['transcript'], model)
            
            y_true.append(row['emotion'])
            y_pred.append(result['emotion'])
            confidences.append(result['confidence'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
                
        except Exception as e:
            print(f"Error processing {row['transcript']}: {e}")
            continue
    
    return y_true, y_pred, confidences


def evaluate_fusion_model(test_df, model):
    """
    Evaluate fusion model
    """
    print("\n" + "="*60)
    print("EVALUATING FUSION MODEL")
    print("="*60)
    
    y_true = []
    y_pred = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        try:
            result = predict_fusion_emotion(
                row['audio_path'],
                row['transcript'],
                model
            )
            
            y_true.append(row['emotion'])
            y_pred.append(result['emotion'])
            confidences.append(result['confidence'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples...")
                
        except Exception as e:
            print(f"Error processing {row['audio_path']}: {e}")
            continue
    
    return y_true, y_pred, confidences


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")


def generate_classification_report(y_true, y_pred, model_name):
    """
    Generate and save classification report
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=EMOTION_LABELS,
        target_names=EMOTION_LABELS,
        digits=4
    )
    
    filename = f"{model_name}_classification_report.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"{model_name.upper()} MODEL CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"Saved: {filename}")
    return report


def create_comparison_table(results_dict):
    """
    Create comparison table for all models
    """
    comparison_data = []
    
    for model_name, (y_true, y_pred, confidences) in results_dict.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=EMOTION_LABELS,
            average='weighted'
        )
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{accuracy:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1-Score': f"{f1:.4f}",
            'Avg Confidence': f"{np.mean(confidences):.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save as CSV
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_comparison.to_csv(csv_path, index=False)
    print(f"Saved: model_comparison.csv")
    
    # Save as formatted text
    txt_path = os.path.join(RESULTS_DIR, "model_comparison.txt")
    with open(txt_path, 'w') as f:
        f.write("MODEL COMPARISON TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(df_comparison.to_string(index=False))
    print(f"Saved: model_comparison.txt")
    
    return df_comparison


def main():
    """
    Main evaluation pipeline
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Step 1: Create metadata
    df = create_metadata_from_dataset()
    
    # Step 2: Create test split
    test_df = create_test_split(df)
    
    # Step 3: Load models
    print("\nLoading models...")
    speech_model = load_speech_model(SPEECH_MODEL_PATH)
    text_model = load_text_model(TEXT_MODEL_PATH)
    fusion_model = load_fusion_model(FUSION_MODEL_PATH)
    print("Models loaded successfully!")
    
    # Step 4: Evaluate each model
    speech_results = evaluate_speech_model(test_df, speech_model)
    text_results = evaluate_text_model(test_df, text_model)
    fusion_results = evaluate_fusion_model(test_df, fusion_model)
    
    # Step 5: Generate confusion matrices
    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRICES")
    print("="*60)
    
    plot_confusion_matrix(
        speech_results[0], speech_results[1],
        "Speech Model Confusion Matrix",
        "speech_confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        text_results[0], text_results[1],
        "Text Model Confusion Matrix",
        "text_confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        fusion_results[0], fusion_results[1],
        "Fusion Model Confusion Matrix",
        "fusion_confusion_matrix.png"
    )
    
    # Step 6: Generate classification reports
    print("\n" + "="*60)
    print("GENERATING CLASSIFICATION REPORTS")
    print("="*60)
    
    generate_classification_report(speech_results[0], speech_results[1], "speech")
    generate_classification_report(text_results[0], text_results[1], "text")
    generate_classification_report(fusion_results[0], fusion_results[1], "fusion")
    
    # Step 7: Create comparison table
    print("\n" + "="*60)
    print("CREATING COMPARISON TABLE")
    print("="*60)
    
    results_dict = {
        'Speech': speech_results,
        'Text': text_results,
        'Fusion': fusion_results
    }
    
    comparison_df = create_comparison_table(results_dict)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("\nComparison Summary:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
