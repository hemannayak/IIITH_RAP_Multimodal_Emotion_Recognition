import torch
import numpy as np
import librosa

from transformers import (
    DistilBertTokenizer
)

from models.fusion_model import (
    MultimodalFusionModel
)


emotion_labels = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "pleasant_surprise",

    "sad"
]


device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


tokenizer = DistilBertTokenizer.from_pretrained(

    "distilbert-base-uncased"
)


def extract_speech_features(audio_path):
    """
    Extract speech features EXACTLY as fusion training
    """
    
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000
    )

    # Trim Silence
    audio, _ = librosa.effects.trim(
        audio
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Delta
    delta_mfcc = librosa.feature.delta(
        mfcc
    )

    # Delta-Delta
    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    combined_features = np.concatenate(
        [
            mfcc,
            delta_mfcc,
            delta2_mfcc
        ],
        axis=0
    )
    
    combined_features = combined_features.T
    
    MAX_LENGTH = 200
    
    if combined_features.shape[0] < MAX_LENGTH:
        padding = MAX_LENGTH - combined_features.shape[0]
        combined_features = np.pad(
            combined_features,
            (
                (0, padding),
                (0, 0)
            ),
            mode='constant'
        )
    else:
        combined_features = combined_features[
            :MAX_LENGTH,
            :
        ]
    
    return combined_features


def load_fusion_model(
    model_path
):
    """
    Load trained fusion model
    """

    model = MultimodalFusionModel().to(
        device
    )

    model.load_state_dict(

        torch.load(

            model_path,

            map_location=device
        )
    )

    model.eval()

    return model


def predict_fusion_emotion(
    audio_path,
    word,
    model
):
    """
    Multimodal fusion prediction
    
    CRITICAL: Use RAW word + max_length=16
    (matching fusion training, NOT text-only training)
    """

    # ==================================================
    # SPEECH FEATURES
    # ==================================================

    speech_features = extract_speech_features(
        audio_path
    )

    speech_tensor = torch.tensor(
        speech_features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # ==================================================
    # TEXT FEATURES
    # ==================================================
    
    # CRITICAL: RAW word, max_length=16
    # (NOT contextual prompting like text-only model)

    encoding = tokenizer(

        word,  # ← RAW word

        padding="max_length",

        truncation=True,

        max_length=16,  # ← 16, not 32

        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(device)

    attention_mask = encoding[
        "attention_mask"
    ].to(device)

    # ==================================================
    # FUSION PREDICTION
    # ==================================================

    with torch.no_grad():

        logits, fusion_embedding = model(

            speech_tensor,

            input_ids,

            attention_mask
        )

        probabilities = torch.softmax(

            logits,

            dim=1
        )

        predicted_idx = torch.argmax(

            probabilities,

            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_idx
        ].item()

    return {

        "emotion":
        emotion_labels[
            predicted_idx
        ],

        "confidence":
        confidence,

        "probabilities":
        probabilities.cpu().numpy(),

        "fusion_embedding":
        fusion_embedding.cpu().numpy()
    }
