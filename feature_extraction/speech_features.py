import librosa
import numpy as np


def extract_mfcc(

    audio,

    sample_rate=22050,

    n_mfcc=120,

    max_length=120
):
    """
    Extract MFCC features
    """

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sample_rate,

        n_mfcc=n_mfcc
    )

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (
        np.std(mfcc) + 1e-8
    )

    # Pad / truncate
    if mfcc.shape[1] < max_length:

        pad_width = (

            max_length

            - mfcc.shape[1]
        )

        mfcc = np.pad(

            mfcc,

            ((0, 0), (0, pad_width))
        )

    else:

        mfcc = mfcc[
            :,
            :max_length
        ]

    return mfcc.T
