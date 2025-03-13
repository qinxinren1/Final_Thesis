import librosa
import numpy as np

# Load the audio file
audio_path = 'output/audio_07/other.wav'
y, sr = librosa.load(audio_path, sr=None)

# ----------------------------
# 1. Beat Separability
# ----------------------------

# Spectral Contrast
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
mean_spectral_contrast = np.mean(spectral_contrast)
print(f"Mean Spectral Contrast: {mean_spectral_contrast}")

# Energy Differentiation
rms_energy = librosa.feature.rms(y=y)
onsets = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
beat_rms = rms_energy[:, onsets]
non_beat_rms = np.delete(rms_energy, onsets, axis=1)
mean_beat_rms = np.mean(beat_rms)
mean_non_beat_rms = np.mean(non_beat_rms)
energy_differentiation = mean_beat_rms - mean_non_beat_rms
print(f"Energy Differentiation: {energy_differentiation}")

# Temporal Clarity
onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
mean_onset_strength = np.mean(onset_strength)
print(f"Mean Onset Strength: {mean_onset_strength}")

# Harmonic-Percussive Separation
y_harmonic, y_percussive = librosa.effects.hpss(y)
percussive_rms = librosa.feature.rms(y=y_percussive)
mean_percussive_rms = np.mean(percussive_rms)
print(f"Mean Percussive RMS Energy: {mean_percussive_rms}")

# ----------------------------
# 2. Rhythmic Clarity
# ----------------------------

# Beat Salience
beat_strength = librosa.onset.onset_strength(y=y, sr=sr)
mean_beat_strength = np.mean(beat_strength)
print(f"Mean Beat Strength: {mean_beat_strength}")