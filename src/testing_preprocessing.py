
from pathlib import Path
import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

files = [
    PROJECT_ROOT / "data" / "samples" / "sample_drone.wav",
    PROJECT_ROOT / "data" / "samples" / "sample_background_noise.wav",
]

for file_path in files:
    print("\nFile:", file_path.name)

    # Load original audio without trimming or normalization
    audio, sr = librosa.load(
        str(file_path),
        sr=16000,
        mono=True
    )

    print("Sample rate:", sr)
    print("Duration:", round(len(audio) / sr, 2), "seconds")
    print("Original maximum amplitude:", round(float(np.max(np.abs(audio))), 6))
    print("Original RMS:", round(float(np.sqrt(np.mean(audio ** 2))), 6))
    print("Is completely silent:", bool(np.all(audio == 0)))