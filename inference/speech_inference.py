import torch
import numpy as np

from preprocessing.speech_preprocessing import (
    load_audio
)

from feature_extraction.speech_features import (
    extract_mfcc
)

from models.speech_model import (
    SpeechEmotionModel
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


def load_speech_model(

    model_path
):
    """
    Load trained speech model
    """

    model = SpeechEmotionModel().to(
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


def predict_speech_emotion(

    file_path,

    model
):
    """
    Predict emotion
    """

    audio = load_audio(
        file_path
    )

    mfcc = extract_mfcc(
        audio
    )

    features = torch.tensor(

        mfcc,

        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(
            features
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
        probabilities.cpu().numpy()
    }
