import librosa

y, sr = librosa.load('drums.wav')
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print("Tempo:", tempo)
print("Beat times:", beat_times)

