from inference.speech_inference import (
    load_speech_model,
    predict_speech_emotion
)

from inference.fusion_inference import (
    load_fusion_model,
    predict_fusion_emotion
)

SPEECH_MODEL_PATH = (
    "saved_models/"
    "advanced_speech_emotion_model.pth"
)

FUSION_MODEL_PATH = (
    "saved_models/"
    "multimodal_fusion_model.pth"
)

# Test samples with audio + word pairs
test_samples = [

    (
        "dataset/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav",
        "back",
        "fear"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_happy/OAF_back_happy.wav",
        "back",
        "happy"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_Sad/OAF_back_sad.wav",
        "back",
        "sad"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav",
        "back",
        "angry"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_disgust/OAF_back_disgust.wav",
        "back",
        "disgust"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_neutral/OAF_back_neutral.wav",
        "back",
        "neutral"
    ),

    (
        "dataset/TESS Toronto emotional speech set data/OAF_Pleasant_surprise/OAF_back_ps.wav",
        "back",
        "pleasant_surprise"
    )

]

print("\n" + "="*60)
print("MULTIMODAL FUSION TESTING")
print("="*60 + "\n")

print("Loading models...")
speech_model = load_speech_model(SPEECH_MODEL_PATH)
fusion_model = load_fusion_model(FUSION_MODEL_PATH)
print("Models loaded successfully!\n")

speech_correct = 0
fusion_correct = 0

for audio_path, word, actual_label in test_samples:

    # Speech-only prediction
    speech_result = predict_speech_emotion(
        audio_path,
        speech_model
    )

    # Fusion prediction
    fusion_result = predict_fusion_emotion(
        audio_path,
        word,
        fusion_model
    )

    speech_match = (
        speech_result["emotion"]
        ==
        actual_label
    )

    fusion_match = (
        fusion_result["emotion"]
        ==
        actual_label
    )

    if speech_match:
        speech_correct += 1

    if fusion_match:
        fusion_correct += 1

    print(f"Actual: {actual_label}")
    print(f"Word: {word}")
    print(f"Speech-only: {speech_result['emotion']} ({speech_result['confidence']:.2%}) {'✅' if speech_match else '❌'}")
    print(f"Fusion: {fusion_result['emotion']} ({fusion_result['confidence']:.2%}) {'✅' if fusion_match else '❌'}")
    print("-" * 60)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Speech-only Accuracy: {speech_correct}/7 ({speech_correct/7*100:.1f}%)")
print(f"Fusion Accuracy: {fusion_correct}/7 ({fusion_correct/7*100:.1f}%)")
print("="*60 + "\n")
