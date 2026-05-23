import librosa
import numpy as np


def load_audio(

    file_path,

    sample_rate=22050,

    duration=3
):
    """
    Load audio and standardize length.
    """

    audio, sr = librosa.load(

        file_path,

        sr=sample_rate
    )

    # Trim silence
    audio, _ = librosa.effects.trim(
        audio
    )

    # Fixed length
    target_length = sample_rate * duration

    if len(audio) > target_length:

        audio = audio[:target_length]

    else:

        padding = target_length - len(audio)

        audio = np.pad(
            audio,
            (0, padding)
        )

    return audio
