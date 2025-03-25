import os
import uuid
import numpy as np
import subprocess
import time
import json
from flask import Flask, render_template, request, send_from_directory, jsonify
from madmom.features import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor
import librosa
import tempfile
import argparse
import mido  # Import mido for MIDI file handling

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': tempfile.gettempdir(),
    'AUDIO_FOLDER': 'static/audio',
    'SEPARATED_FOLDER': 'static/separated',
    'MIDI_FOLDER': 'static/midi',  # Add MIDI folder
    'ALLOWED_EXTENSIONS': {'wav', 'mp3', 'mid'},  # Add 'mid' as allowed extension
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024
})

os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEPARATED_FOLDER'], exist_ok=True)
os.makedirs(app.config['MIDI_FOLDER'], exist_ok=True)  # Create MIDI folder

def analyze_audio(filepath):
    # Process audio with Madmom
    beat_processor = DBNBeatTrackingProcessor(fps=100)
    act_processor = RNNBeatProcessor()(filepath)
    beats = beat_processor(act_processor)
    
    # Convert beat times to integer indices (multiply by fps to get frame indices)
    beat_indices = (beats * 100).astype(int)  # fps is 100
    
    # Get beat activation values for strength
    beat_activations = act_processor[beat_indices]
    
    # Normalize activations to 0-1 range
    beat_activations = (beat_activations - np.min(beat_activations)) / (np.max(beat_activations) - np.min(beat_activations))
    
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
    
    # Create measures and beats
    measures = []
    current_measure = 1
    beat_count = 0
    current_measure_beats = []
    
    # Get duration from librosa
    y, sr = librosa.load(filepath, sr=44100, mono=True)
    duration = round(float(librosa.get_duration(y=y, sr=sr)), 2)
    
    for i, beat in enumerate(beats):
        # Add beat information
        beat_info = {
            'time': round(float(beat), 2),
            'strength': round(float(beat_activations[i]), 2),
            'is_strong': bool(beat_activations[i] > 0.7)  # Convert NumPy bool_ to Python bool
        }
        
        # Start a new measure if we've reached the beat count for the current measure
        if beat_count % best_denominator == 0:
            if current_measure_beats:  # If we have beats from the previous measure
                measures.append({
                    'number': current_measure,
                    'start': current_measure_beats[0]['time'],
                    'end': round(float(beat), 2),
                    'beats': current_measure_beats.copy()  # Make a copy of the beats for this measure
                })
                current_measure += 1
            current_measure_beats = []  # Reset beats for next measure
        
        current_measure_beats.append(beat_info)
        beat_count += 1
    
    # Add the last measure if it has any beats
    if current_measure_beats:
        measures.append({
            'number': current_measure,
            'start': current_measure_beats[0]['time'],
            'end': duration,
            'beats': current_measure_beats
        })
    
    return {
        'tempo': tempo,
        'time_sig': f"4/{best_denominator}",
        'measures': measures,
        'duration': duration,
        'beats': [beat for measure in measures for beat in measure['beats']]  # Flatten beats for overall list
    }

def analyze_midi(filepath):
    """Analyze a MIDI file and extract track information"""
    midi_data = mido.MidiFile(filepath)
    
    # Calculate duration in seconds
    duration = midi_data.length
    
    # Extract tempo if available
    tempo = 120  # Default tempo if not specified
    for track in midi_data.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                # Convert microseconds per beat to BPM
                tempo = int(round(60000000 / msg.tempo))
                break
        if tempo != 120:  # Stop if we found a tempo
            break
    
    # Extract time signature if available
    numerator = 4
    denominator = 4
    for track in midi_data.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                numerator = msg.numerator
                denominator = msg.denominator
                break
        if numerator != 4 or denominator != 4:  # Stop if we found a time signature
            break
    
    # Extract tracks information
    tracks = []
    for i, track in enumerate(midi_data.tracks):
        track_info = {
            'number': i,
            'name': track.name if hasattr(track, 'name') and track.name else f'Track {i}',
            'notes_count': sum(1 for msg in track if msg.type == 'note_on'),
            'program_changes': [msg for msg in track if msg.type == 'program_change'],
            'instruments': []
        }
        
        # Get unique instruments in the track
        instruments = set()
        current_program = 0
        for msg in track:
            if msg.type == 'program_change':
                current_program = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                instruments.add(current_program)
        
        # Convert instrument numbers to names
        for program in instruments:
            # Simple mapping of common MIDI programs to instrument names
            if 0 <= program <= 7:
                instrument = "Piano"
            elif 8 <= program <= 15:
                instrument = "Chromatic Percussion"
            elif 16 <= program <= 23:
                instrument = "Organ"
            elif 24 <= program <= 31:
                instrument = "Guitar"
            elif 32 <= program <= 39:
                instrument = "Bass"
            elif 40 <= program <= 47:
                instrument = "Strings"
            elif 48 <= program <= 55:
                instrument = "Ensemble"
            elif 56 <= program <= 63:
                instrument = "Brass"
            elif 64 <= program <= 71:
                instrument = "Reed"
            elif 72 <= program <= 79:
                instrument = "Pipe"
            elif 80 <= program <= 87:
                instrument = "Synth Lead"
            elif 88 <= program <= 95:
                instrument = "Synth Pad"
            elif 96 <= program <= 103:
                instrument = "Synth Effects"
            elif 104 <= program <= 111:
                instrument = "Ethnic"
            elif 112 <= program <= 119:
                instrument = "Percussive"
            elif 120 <= program <= 127:
                instrument = "Sound Effects"
            else:
                instrument = "Unknown"
                
            track_info['instruments'].append({
                'program': program,
                'name': instrument
            })
        
        tracks.append(track_info)
    
    return {
        'format': midi_data.type,
        'tracks_count': len(midi_data.tracks),
        'tempo': tempo,
        'time_sig': f"{numerator}/{denominator}",
        'duration': round(duration, 2),
        'tracks': tracks
    }

def submit_separation_job(audio_path, job_id):
    """Submit a track separation job to DelftBlue or run locally if sbatch is not available"""
    output_dir = os.path.join(app.config['SEPARATED_FOLDER'], job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the separation script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'separate_tracks.sh')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    # First, check if we're on DelftBlue by testing if sbatch is available
    try:
        subprocess.run(['which', 'sbatch'], check=True, capture_output=True, text=True)
        use_slurm = True
    except subprocess.CalledProcessError:
        use_slurm = False
    
    job_info = {
        'job_id': job_id,
        'status': 'submitted',
        'audio_path': audio_path,
        'output_dir': output_dir,
        'submit_time': time.time(),
        'use_slurm': use_slurm
    }
    
    if use_slurm:
        # Convert relative path to absolute path for DelftBlue
        abs_audio_path = os.path.abspath(audio_path)
        abs_output_dir = os.path.abspath(output_dir)
        
        # Submit the job to SLURM
        cmd = ['sbatch', script_path, abs_audio_path, abs_output_dir, job_id]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Extract job ID from SLURM output (usually "Submitted batch job 12345")
            slurm_job_id = result.stdout.strip().split()[-1]
            job_info['slurm_job_id'] = slurm_job_id
            print(f"Successfully submitted job {slurm_job_id} for {job_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            job_info['status'] = 'error'
            job_info['error_message'] = f"STDOUT: {e.stdout}, STDERR: {e.stderr}"
            
            # Create an error log file for debugging
            with open(os.path.join(output_dir, f"{job_id}_error.log"), 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"STDOUT: {e.stdout}\n")
                f.write(f"STDERR: {e.stderr}\n")
    else:
        # Run locally in a separate process
        print("SLURM not available, running separation locally in background")
        job_info['status'] = 'running'
        
        # Create a simple script to run locally
        local_script = f"""#!/bin/bash
echo "Starting local separation for file: {audio_path}"
echo "Output directory: {output_dir}"

# Run demucs if available, otherwise just create dummy files for testing
if command -v demucs &> /dev/null; then
    python -m demucs.separate -n htdemucs "{audio_path}" -o "{output_dir}"
else
    # Create dummy files for testing
    mkdir -p "{output_dir}/htdemucs/test"
    touch "{output_dir}/htdemucs/test/drums.wav"
    touch "{output_dir}/htdemucs/test/bass.wav"
    touch "{output_dir}/htdemucs/test/vocals.wav"
    touch "{output_dir}/htdemucs/test/other.wav"
fi

echo "completed" > "{output_dir}/{job_id}_status.txt"
echo "Separation completed"
"""
        local_script_path = os.path.join(output_dir, f"{job_id}_local.sh")
        with open(local_script_path, 'w') as f:
            f.write(local_script)
        os.chmod(local_script_path, 0o755)
        
        # Run in background
        subprocess.Popen(
            ['/bin/bash', local_script_path], 
            stdout=open(os.path.join(output_dir, f"{job_id}_local.log"), 'w'),
            stderr=subprocess.STDOUT
        )
    
    # Save job info
    with open(os.path.join(output_dir, f"{job_id}_info.json"), 'w') as f:
        json.dump(job_info, f)
        
    return job_info

def check_job_status(job_id):
    """Check the status of a separation job"""
    output_dir = os.path.join(app.config['SEPARATED_FOLDER'], job_id)
    status_file = os.path.join(output_dir, f"{job_id}_status.txt")
    error_log = os.path.join(output_dir, f"{job_id}_error.log")
    
    # Check for completed status
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = f.read().strip()
        return {'status': status}
    
    # Check for error log
    if os.path.exists(error_log):
        with open(error_log, 'r') as f:
            error_content = f.read()
        return {'status': 'error', 'error_details': error_content}
    
    # If status file doesn't exist, check if the job info exists
    info_file = os.path.join(output_dir, f"{job_id}_info.json")
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            job_info = json.load(f)
        
        # If job already has error status, return it with details
        if job_info.get('status') == 'error':
            return {
                'status': 'error', 
                'error_details': job_info.get('error_message', 'Unknown error')
            }
        
        # Check if using SLURM
        if job_info.get('use_slurm', False) and 'slurm_job_id' in job_info:
            # Check SLURM job status
            slurm_job_id = job_info['slurm_job_id']
            cmd = ['squeue', '-j', slurm_job_id, '-h']
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stdout.strip():
                    # Job is in the queue, check its state
                    state = result.stdout.strip().split()[4] if len(result.stdout.strip().split()) > 4 else "UNKNOWN"
                    if state == "PD":
                        return {'status': 'pending', 'details': 'Job is pending in the queue'}
                    elif state == "R":
                        return {'status': 'running', 'details': 'Job is running'}
                    else:
                        return {'status': 'queued', 'details': f'Job state: {state}'}
                else:
                    # Job not in queue, check if it completed without creating a status file
                    sacct_cmd = ['sacct', '-j', slurm_job_id, '--format=State', '--noheader']
                    try:
                        sacct_result = subprocess.run(sacct_cmd, check=True, capture_output=True, text=True)
                        state = sacct_result.stdout.strip().split()[0] if sacct_result.stdout.strip() else "UNKNOWN"
                        if state == "COMPLETED":
                            # Job completed but status file wasn't created, create it now
                            with open(status_file, 'w') as f:
                                f.write("completed")
                            return {'status': 'completed'}
                        elif state in ["FAILED", "TIMEOUT", "CANCELLED"]:
                            return {'status': 'error', 'error_details': f'SLURM job {slurm_job_id} ended with state {state}'}
                        else:
                            return {'status': 'unknown', 'details': f'SLURM job state: {state}'}
                    except:
                        # Can't determine job state from sacct
                        return {'status': 'unknown', 'details': 'Job not in queue and state unknown'}
            except:
                return {'status': 'unknown', 'details': 'Error checking SLURM job status'}
        else:
            # For local jobs, just return the status from job_info
            return {'status': job_info.get('status', 'unknown')}
    
    return {'status': 'not_found'}

def get_separated_tracks(job_id):
    """Get the paths to separated tracks"""
    output_dir = os.path.join(app.config['SEPARATED_FOLDER'], job_id)
    
    # Check for the htdemucs output structure
    model_dir = os.path.join(output_dir, 'htdemucs')
    if not os.path.exists(model_dir):
        return None
    
    # Find the audio file directory (named after the input file)
    audio_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not audio_dirs:
        return None
    
    audio_dir = os.path.join(model_dir, audio_dirs[0])
    
    # Get the separated tracks
    tracks = {}
    for track in os.listdir(audio_dir):
        if track.endswith('.wav'):
            track_name = os.path.splitext(track)[0]
            tracks[track_name] = f"/separated/{job_id}/htdemucs/{audio_dirs[0]}/{track}"
    
    # If no tracks found but we have a status file indicating completion,
    # create dummy tracks for testing
    if not tracks and os.path.exists(os.path.join(output_dir, f"{job_id}_status.txt")):
        # Create dummy files for testing UI
        dummy_dir = os.path.join(model_dir, 'test')
        os.makedirs(dummy_dir, exist_ok=True)
        
        # Create empty files if they don't exist
        for dummy_track in ['drums', 'bass', 'vocals', 'other']:
            dummy_file = os.path.join(dummy_dir, f"{dummy_track}.wav")
            if not os.path.exists(dummy_file):
                with open(dummy_file, 'w') as f:
                    f.write('')
            tracks[dummy_track] = f"/separated/{job_id}/htdemucs/test/{dummy_track}.wav"
    
    return tracks

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis = None
    error = None
    audio_url = None
    separation_job_id = None
    is_midi = False

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file uploaded'
        else:
            file = request.files['file']
            if file and file.filename.split('.')[-1].lower() in app.config['ALLOWED_EXTENSIONS']:
                ext = file.filename.rsplit('.', 1)[1].lower()
                uid = uuid.uuid4().hex
                filename = f"{uid}.{ext}"
                
                if ext == 'mid':
                    # Handle MIDI file
                    save_path = os.path.join(app.config['MIDI_FOLDER'], filename)
                    try:
                        file.save(save_path)
                        analysis = analyze_midi(save_path)
                        analysis['midi_url'] = f"/midi/{filename}"
                        analysis['is_midi'] = True
                        is_midi = True
                    except Exception as e:
                        error = f"MIDI processing error: {str(e)}"
                        if os.path.exists(save_path):
                            os.remove(save_path)
                else:
                    # Handle audio file (existing functionality)
                    save_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
                    try:
                        file.save(save_path)
                        analysis = analyze_audio(save_path)
                        analysis['audio_url'] = f"/audio/{filename}"
                        
                        # Submit separation job if requested
                        if request.form.get('separate_tracks') == 'true':
                            separation_job_id = f"sep_{uid}"
                            job_info = submit_separation_job(save_path, separation_job_id)
                            analysis['separation_job_id'] = separation_job_id
                            analysis['separation_status'] = job_info['status']
                    except Exception as e:
                        error = f"Processing error: {str(e)}"
                        if os.path.exists(save_path):
                            os.remove(save_path)
            else:
                error = 'Invalid file format'

    return render_template('index.html', analysis=analysis, error=error, separation_job_id=separation_job_id, is_midi=is_midi)

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

@app.route('/separated/<path:filename>')
def serve_separated(filename):
    # Split the path to get the directory and file
    parts = filename.split('/')
    directory = os.path.join(app.config['SEPARATED_FOLDER'], *parts[:-1])
    file = parts[-1]
    return send_from_directory(directory, file)

@app.route('/check_separation/<job_id>')
def check_separation(job_id):
    status = check_job_status(job_id)
    
    if status['status'] == 'completed':
        tracks = get_separated_tracks(job_id)
        if tracks:
            return json.dumps({'status': 'completed', 'tracks': tracks}, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
    
    return json.dumps(status, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

@app.route('/analyze_track/<job_id>/<track_name>')
def analyze_track(job_id, track_name):
    """Analyze a specific separated track"""
    # Get the separated tracks
    tracks = get_separated_tracks(job_id)
    if not tracks or track_name not in tracks:
        return jsonify({'error': 'Track not found'})
    
    # Get the full path to the track
    track_url = tracks[track_name]
    track_path = os.path.join(app.config['SEPARATED_FOLDER'], *track_url.split('/')[2:])
    
    if not os.path.exists(track_path):
        return jsonify({'error': 'Track file not found'})
    
    try:
        # Analyze the track
        analysis = analyze_audio(track_path)
        analysis['track_name'] = track_name
        analysis['audio_url'] = track_url
        return json.dumps(analysis, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/midi/<filename>')
def serve_midi(filename):
    return send_from_directory(app.config['MIDI_FOLDER'], filename)

@app.route('/midi_track/<filename>/<int:track_num>')
def get_midi_track(filename, track_num):
    """Extract a specific track from a MIDI file and return as a new MIDI file and note data"""
    filepath = os.path.join(app.config['MIDI_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'MIDI file not found'}), 404
    
    try:
        midi_data = mido.MidiFile(filepath)
        
        if track_num >= len(midi_data.tracks):
            return jsonify({'error': 'Track not found'}), 404
        
        # Create a new MIDI file with just this track
        output_midi = mido.MidiFile(type=0)  # Type 0 means single track
        output_midi.ticks_per_beat = midi_data.ticks_per_beat
        
        # Copy the specified track
        output_midi.tracks.append(midi_data.tracks[track_num])
        
        # Save to a temporary file
        track_uid = uuid.uuid4().hex
        output_filename = f"{track_uid}_track_{track_num}.mid"
        output_path = os.path.join(app.config['MIDI_FOLDER'], output_filename)
        output_midi.save(output_path)
        
        # Extract note data for browser playback
        # This is a simplified representation of MIDI notes
        notes = []
        current_time = 0
        tempo = 500000  # Default tempo (microseconds per beat)
        ticks_per_beat = midi_data.ticks_per_beat
        
        # Look for common instrument
        instrument = "acoustic_grand_piano"  # Default instrument
        for msg in midi_data.tracks[track_num]:
            if msg.type == 'program_change':
                # Map program number to general MIDI instrument name
                program = msg.program
                if 0 <= program <= 7:
                    instrument = "acoustic_grand_piano"
                elif 8 <= program <= 15:
                    instrument = "glockenspiel"
                elif 16 <= program <= 23:
                    instrument = "church_organ"
                elif 24 <= program <= 31:
                    instrument = "acoustic_guitar_nylon"
                elif 32 <= program <= 39:
                    instrument = "acoustic_bass"
                elif 40 <= program <= 47:
                    instrument = "violin"
                elif 48 <= program <= 55:
                    instrument = "string_ensemble_1"
                elif 56 <= program <= 63:
                    instrument = "trumpet"
                elif 64 <= program <= 71:
                    instrument = "alto_sax"
                elif 72 <= program <= 79:
                    instrument = "flute"
                elif 80 <= program <= 87:
                    instrument = "lead_1_square"
                elif 88 <= program <= 95:
                    instrument = "pad_2_warm"
                elif 96 <= program <= 103:
                    instrument = "fx_1_rain"
                elif 104 <= program <= 111:
                    instrument = "sitar"
                elif 112 <= program <= 119:
                    instrument = "woodblock"
                elif 120 <= program <= 127:
                    instrument = "bird_tweet"
                break
        
        # Track note on/offs to pair them
        active_notes = {}
        
        for msg in midi_data.tracks[track_num]:
            if not msg.is_meta:
                current_time += msg.time
                
                # Handle tempo changes
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
                # Track note on/off events
                elif msg.type == 'note_on' and msg.velocity > 0:
                    # Note start
                    seconds = mido.tick2second(current_time, ticks_per_beat, tempo)
                    active_notes[msg.note] = {
                        'note': msg.note,
                        'startTime': seconds,
                        'velocity': msg.velocity / 127.0
                    }
                    
                elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note end - find matching note_on and calculate duration
                    if msg.note in active_notes:
                        seconds = mido.tick2second(current_time, ticks_per_beat, tempo)
                        note_data = active_notes[msg.note]
                        note_data['duration'] = seconds - note_data['startTime']
                        notes.append(note_data)
                        del active_notes[msg.note]
        
        return jsonify({
            'track_num': track_num,
            'track_url': f"/midi/{output_filename}",
            'notes': notes,
            'instrument': instrument
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask music analysis web application')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on (default: 8080)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on (default: 127.0.0.1)')
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port)