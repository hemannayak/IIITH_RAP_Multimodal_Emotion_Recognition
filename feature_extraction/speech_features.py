import librosa
import numpy as np


N_MFCC = 40
MAX_PAD_LENGTH = 200


def extract_mfcc(
    signal,
    sample_rate=22050
):
    """
    Match training feature extraction EXACTLY
    """

    # MFCC
    mfcc = librosa.feature.mfcc(

        y=signal,

        sr=sample_rate,

        n_mfcc=N_MFCC
    )

    # Delta
    delta_mfcc = librosa.feature.delta(

        mfcc
    )

    # Delta²
    delta2_mfcc = librosa.feature.delta(

        mfcc,

        order=2
    )

    # Combine features
    combined_features = np.vstack([

        mfcc,

        delta_mfcc,

        delta2_mfcc
    ])

    # Padding / truncation
    if combined_features.shape[1] < MAX_PAD_LENGTH:

        pad_width = (

            MAX_PAD_LENGTH

            - combined_features.shape[1]
        )

        combined_features = np.pad(

            combined_features,

            ((0, 0), (0, pad_width)),

            mode="constant"
        )

    else:

        combined_features = combined_features[
            :,
            :MAX_PAD_LENGTH
        ]

    return combined_features.T
