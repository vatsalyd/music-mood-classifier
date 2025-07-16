import sys
import os
import librosa
import numpy as np
import pandas as pd
import joblib

model = joblib.load('./models/music_mood_classifier.pkl')
print("✅ Model loaded!")

def extract_features(filepath):
    y, sr = librosa.load(filepath, duration=30)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    mfccs_mean = np.mean(mfccs, axis=1)

    features = {
        'tempo': tempo,
        'spectral_centroid': spectral_centroid,
        'chroma': chroma,
        'zcr': zcr
    }

    for i, mfcc in enumerate(mfccs_mean):
        features[f'mfcc_{i+1}'] = mfcc

    return features

if len(sys.argv) != 2:
    print("Usage: python predict.py <path_to_audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

if not os.path.isfile(audio_file):
    print(f"❌ File not found: {audio_file}")
    sys.exit(1)

print(f"🎵 Extracting features for: {audio_file}")


features = extract_features(audio_file)
X_new = pd.DataFrame([features])


mood = model.predict(X_new)[0]
print(f"✅ Predicted Mood: {mood}")
