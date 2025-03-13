import os
import uuid
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from madmom.features import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
import librosa
import tempfile

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': tempfile.gettempdir(),
    'AUDIO_FOLDER': 'static/audio',
    'ALLOWED_EXTENSIONS': {'wav', 'mp3'},
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024
})

os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

def analyze_audio(filepath):
    # Process audio with Madmom
    beat_processor = DBNBeatTrackingProcessor(fps=100)
    act_processor = RNNBeatProcessor()(filepath)
    beats = beat_processor(act_processor)
    
    # Estimate tempo
    tempo_processor = TempoEstimationProcessor(fps=100)(act_processor)
    tempo = int(round(tempo_processor[np.argmax(tempo_processor[:, 1])][0]))
    
    # Detect time signature
    intervals = np.diff(beats)
    best_denominator = 4
    if len(intervals) > 3:
        median_interval = np.median(intervals)
        scores = []
        for denom in [2, 3, 4, 6, 8]:
            scores.append(np.sum(np.abs(intervals - (median_interval * denom))))
        best_denominator = [2, 3, 4, 6, 8][np.argmin(scores)]
    
    # Create measures
    measures = []
    current_measure = 1
    beat_count = 0
    
    for beat in beats:
        if beat_count % best_denominator == 0:
            if measures:
                measures[-1]['end'] = round(float(beat), 2)
            measures.append({
                'number': current_measure,
                'start': round(float(beat), 2),
                'end': None
            })
            current_measure += 1
        beat_count += 1
    
    # Get duration from librosa
    y, sr = librosa.load(filepath, sr=44100, mono=True)
    duration = round(float(librosa.get_duration(y=y, sr=sr)), 2)
    
    if measures:
        measures[-1]['end'] = duration
    
    return {
        'tempo': tempo,
        'time_sig': f"4/{best_denominator}",
        'measures': measures,
        'duration': duration
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis = None
    error = None
    audio_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file uploaded'
        else:
            file = request.files['file']
            if file and file.filename.split('.')[-1].lower() in app.config['ALLOWED_EXTENSIONS']:
                ext = file.filename.rsplit('.', 1)[1].lower()
                uid = uuid.uuid4().hex
                filename = f"{uid}.{ext}"
                save_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
                
                try:
                    file.save(save_path)
                    analysis = analyze_audio(save_path)
                    analysis['audio_url'] = f"/audio/{filename}"
                except Exception as e:
                    error = f"Processing error: {str(e)}"
                    if os.path.exists(save_path):
                        os.remove(save_path)
            else:
                error = 'Invalid file format'

    return render_template('index.html', analysis=analysis, error=error)

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)