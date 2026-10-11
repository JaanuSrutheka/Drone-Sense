
import librosa
import numpy as np


def preprocess_audio(file_path, target_sr=16000):
    """
    Load and prepare an audio file for DroneSense.

    Parameters:
        file_path: Path to the audio file.
        target_sr: Required sample rate (default 16000 Hz).

    Returns:
        audio: Processed mono audio signal.
        target_sr: Sample rate of the processed signal.
    """

    # Step 1: Load audio, convert to mono, and resample
    audio, sr = librosa.load(
        file_path,
        sr=target_sr,
        mono=True
    )

    # Step 2: Check whether audio contains samples
    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Step 3: Remove leading and trailing silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Step 4: Check again after trimming
    if len(audio) == 0:
        raise ValueError("No usable audio remains after trimming.")

    # Step 5: Normalize amplitude safely
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude > 0:
        audio = audio / max_amplitude

    return audio, target_sr