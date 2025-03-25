class MIDIHandler {
    constructor() {
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('file').addEventListener('change', (e) => this.handleFileChange(e));
    }

    handleFileChange(e) {
        const file = e.target.files[0];
        if (!file) return;

        const fileExt = file.name.split('.').pop().toLowerCase();
        const isMidi = fileExt === 'mid' || fileExt === 'midi';
        
        document.getElementById('midiContainer').style.display = isMidi ? 'block' : 'none';
        document.getElementById('audioSubmitContainer').style.display = isMidi ? 'none' : 'block';

        if (isMidi) {
            this.processMIDIFile(file);
        }
    }

    async processMIDIFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/midi_upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            this.displayMidiInfo(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing MIDI file');
        }
    }

    displayMidiInfo(data) {
        // Display basic information
        document.getElementById('midiTempo').textContent = `${data.basic_info.tempo} BPM`;
        document.getElementById('midiTimeSignature').textContent = data.basic_info.time_signature;
        document.getElementById('midiDuration').textContent = `${data.basic_info.duration} seconds`;

        // Display tracks
        this.displayTracks(data);
    }

    displayTracks(data) {
        const trackList = document.getElementById('trackList');
        trackList.innerHTML = '';
        trackList.dataset.fileId = data.file_id;

        data.tracks.forEach(track => {
            const trackDiv = this.createTrackElement(track);
            trackList.appendChild(trackDiv);
        });
    }

    createTrackElement(track) {
        const trackDiv = document.createElement('div');
        trackDiv.className = 'track-item';
        trackDiv.innerHTML = this.getTrackHTML(track);
        return trackDiv;
    }

    getTrackHTML(track) {
        return `
            <div class="track-header">
                <div class="track-info">
                    <div class="form-check">
                        <input type="checkbox" class="form-check-input" id="track_${track.id}" value="${track.id}">
                        <label class="form-check-label" for="track_${track.id}">
                            <div class="track-name">${track.name}</div>
                            <div class="track-details">
                                ${track.instrument_name}${track.is_drum ? ' (Drums)' : ''}
                                • ${track.note_count} notes
                            </div>
                        </label>
                    </div>
                </div>
            </div>
            ${track.waveform ? `
                <div class="waveform-container">
                    <img src="${track.waveform}" alt="Track waveform" class="waveform-image">
                </div>
            ` : ''}
        `;
    }
}

// Initialize the handler when the document is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MIDIHandler();
}); 