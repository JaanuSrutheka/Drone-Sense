
from pathlib import Path

from src.preprocessing import preprocess_audio
from src.feature_extraction import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent

audio_files = [
    PROJECT_ROOT / "data" / "samples" / "sample_drone.wav",
    PROJECT_ROOT / "data" / "samples" / "sample_background_noise.wav",
]

for file_path in audio_files:
    print("\nFile:", file_path.name)

    audio, sr = preprocess_audio(str(file_path))
    features = extract_features(audio, sr)

    print("Sample rate:", sr)
    print("Feature vector shape:", features.shape)
    print("Number of features:", len(features))
    print("All features finite:", bool(__import__("numpy").isfinite(features).all()))
    print("Feature values:")
    print(features)