
import librosa
import numpy as np


def extract_features(audio, sr):
    """
    Extract numerical audio features for DroneSense.

    Parameters:
        audio: Preprocessed audio signal.
        sr: Sample rate of the audio.

    Returns:
        A NumPy array containing 31 features.
    """

    features = []

    # 1. MFCC features: 13 mean values
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )
    features.extend(np.mean(mfcc, axis=1))

    # 2. MFCC features: 13 standard deviations
    features.extend(np.std(mfcc, axis=1))

    # 3. Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )
    features.append(np.mean(centroid))

    # 4. Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )
    features.append(np.mean(bandwidth))

    # 5. Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )
    features.append(np.mean(rolloff))

    # 6. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.append(np.mean(zcr))

    # 7. Root mean square energy
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))

    return np.array(features, dtype=float)