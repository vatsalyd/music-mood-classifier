import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile


model = joblib.load('./models/music_mood_classifier.pkl')
st.title("🎵 Music Mood Classifier")
st.write("Upload an audio file (WAV/MP3) and I will predict its mood!")


uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

if uploaded_file is not None:
   
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:    # Saved the file to a temporary location
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')

   
    y, sr = librosa.load(file_path, duration=30)

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
        'zcr': zcr,
    }

    for i, mfcc in enumerate(mfccs_mean):
        features[f'mfcc_{i+1}'] = mfcc

    X_new = pd.DataFrame([features])
    mood = model.predict(X_new)[0]

    st.success(f"✅ Predicted Mood: **{mood}**")

