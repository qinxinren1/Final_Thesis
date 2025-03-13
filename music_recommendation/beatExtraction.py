import numpy as np
import librosa
import soundfile as sf

# Load the audio file
audio_path = 'output/audio_07/other.wav'
y, sr = librosa.load(audio_path, sr=None)

# Compute the onset strength envelope
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Set a threshold for strong onsets. 
# Adjust the multiplier (e.g., 1.5) to control sensitivity.
threshold = 1.5 * np.median(onset_env)

# Detect onsets using the customized threshold. 
# This ensures that only onsets with a strength significantly above the median are detected.
onset_times = librosa.onset.onset_detect(
    y=y, 
        sr=sr, 
        units="time", 
        pre_max=3, 
        post_max=3, 
        pre_avg=3, 
        post_avg=5, 
        delta=0.25, 
        wait=0.1
)

print("Detected strong onsets (in seconds):")
print(onset_times)

# Optional: Generate a beep sound (1 kHz, 50 ms) to mark the detected onsets
duration_beep = 0.05  # seconds
t = np.linspace(0, duration_beep, int(sr * duration_beep), endpoint=False)
beep = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1 kHz sine wave, amplitude 0.5

# Mix the beep into the original audio at each detected onset
y_out = y.copy()
onset_samples = (onset_times * sr).astype(int)
for sample in onset_samples:
    end_sample = sample + len(beep)
    if end_sample < len(y_out):
        y_out[sample:end_sample] += beep
    else:
        y_out[sample:] += beep[:len(y_out) - sample]

# Normalize to avoid clipping if necessary
max_val = np.max(np.abs(y_out))
if max_val > 1:
    y_out = y_out / max_val

# Save the new audio file with audible onset markers
output_file = 'other_with_strong_onsets.wav'
sf.write(output_file, y_out, sr)
print(f"Processed audio saved as '{output_file}'.")
