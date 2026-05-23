from inference.text_inference import (
    load_text_model,
    predict_text_emotion
)

MODEL_PATH = (
    "saved_models/"
    "text_emotion_model.pth"
)

model = load_text_model(
    MODEL_PATH
)

test_words = [

    "back",
    "bar",
    "base",
    "bath",
    "chair",
    "juice",
    "dog"
]

print("\nText Model Testing\n")

for word in test_words:

    result = predict_text_emotion(
        word,
        model
    )

    print(
        f"Word: {word}"
    )

    print(
        f"Prediction:"
        f" {result['emotion']}"
    )

    print(
        f"Confidence:"
        f" {result['confidence']:.2%}"
    )

    print("-" * 40)
