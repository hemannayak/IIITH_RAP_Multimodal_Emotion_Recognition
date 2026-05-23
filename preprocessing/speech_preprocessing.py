import librosa


SAMPLE_RATE = 22050


def load_audio(
    file_path
):
    """
    Load and preprocess audio
    """

    signal, sample_rate = librosa.load(

        file_path,

        sr=SAMPLE_RATE
    )

    # Remove silence
    signal, _ = librosa.effects.trim(

        signal
    )

    # Normalize raw signal
    signal = librosa.util.normalize(

        signal
    )

    return signal
