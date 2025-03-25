from pretty_midi import PrettyMIDI
import mido
from ..utils.midi_utils import midi_track_to_audio
from ..utils.visualization import create_waveform_image

class MIDIAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.midi_file = mido.MidiFile(filepath)
        self.pm = PrettyMIDI(filepath)

    def get_essential_info(self):
        """Extract essential MIDI information"""
        try:
            initial_tempo = 120.0
            initial_time_sig = "4/4"
            
            for msg in mido.merge_tracks(self.midi_file.tracks):
                if msg.type == 'set_tempo':
                    initial_tempo = 60000000 / msg.tempo
                    break
                    
            if self.pm.time_signature_changes:
                first_ts = self.pm.time_signature_changes[0]
                initial_time_sig = f"{first_ts.numerator}/{first_ts.denominator}"
            
            main_tracks = self._process_tracks()
            
            return {
                'status': 'success',
                'basic_info': {
                    'tempo': round(initial_tempo, 1),
                    'time_signature': initial_time_sig,
                    'duration': round(self.pm.get_end_time(), 2),
                    'total_tracks': len(main_tracks)
                },
                'tracks': main_tracks
            }
        except Exception as e:
            print(f"Error extracting MIDI information: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _process_tracks(self):
        """Process individual tracks"""
        main_tracks = []
        for i, (track, instrument) in enumerate(zip(self.midi_file.tracks, self.pm.instruments)):
            track_info = self._process_single_track(i, track, instrument)
            if track_info:
                main_tracks.append(track_info)
        return main_tracks

    def _process_single_track(self, track_idx, track, instrument):
        """Process a single track"""
        track_name = f"Track {track_idx + 1}"
        note_count = len(instrument.notes)
        
        if note_count == 0:
            return None

        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
                break
        
        audio_data = midi_track_to_audio(self.pm, track_idx)
        waveform_image = None
        if audio_data is not None:
            waveform_image = create_waveform_image(audio_data)
        
        if waveform_image is None:
            return None

        return {
            'id': track_idx,
            'name': track_name,
            'instrument_name': pretty_midi.program_to_instrument_name(instrument.program) if not instrument.is_drum else 'Drums',
            'is_drum': instrument.is_drum,
            'note_count': note_count,
            'waveform': waveform_image
        } 