import torch

from transformers import (
    DistilBertTokenizer
)

from models.text_model import (
    TextEmotionModel
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


def load_text_model(
    model_path
):
    """
    Load trained text model
    """

    model = TextEmotionModel().to(
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


def predict_text_emotion(
    word,
    model
):
    """
    Match training preprocessing EXACTLY
    """

    # CRITICAL FIX:
    # same prompt as training

    transcript = (

        "The speaker emotionally "

        "expressed the word "

        f"{word}"
    )

    encoding = tokenizer(

        transcript,

        padding="max_length",

        truncation=True,

        max_length=32,

        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(device)

    attention_mask = encoding[
        "attention_mask"
    ].to(device)

    with torch.no_grad():

        logits = model(

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
        probabilities.cpu().numpy()
    }
