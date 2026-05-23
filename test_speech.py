from inference.speech_inference import (
    load_speech_model,
    predict_speech_emotion
)

MODEL_PATH = (
    "saved_models/"
    "advanced_speech_emotion_model.pth"
)

test_files = [

    (
        "dataset/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav",
        "fear"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_happy/OAF_back_happy.wav",
        "happy"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_Sad/OAF_back_sad.wav",
        "sad"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav",
        "angry"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_disgust/OAF_back_disgust.wav",
        "disgust"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_neutral/OAF_back_neutral.wav",
        "neutral"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_Pleasant_surprise/"
        "OAF_back_ps.wav",
        "pleasant_surprise"
    )

]

model = load_speech_model(
    MODEL_PATH
)

correct = 0

print("\nSpeech Model Testing\n")

for audio_path, actual_label in test_files:

    result = predict_speech_emotion(
        audio_path,
        model
    )

    predicted = result["emotion"]

    confidence = result[
        "confidence"
    ]

    is_correct = (
        predicted
        ==
        actual_label
    )

    if is_correct:
        correct += 1

    print(
        f"Actual: {actual_label}"
    )

    print(
        f"Predicted: {predicted}"
    )

    print(
        f"Confidence:"
        f" {confidence:.2%}"
    )

    print(
        f"Correct:"
        f" {'YES' if is_correct else 'NO'}"
    )

    print("-" * 40)

accuracy = (
    correct
    /
    len(test_files)
)

print(
    f"\nFinal Accuracy:"
    f" {accuracy:.2%}"
)
