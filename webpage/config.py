import os
import tempfile

class Config:
    UPLOAD_FOLDER = tempfile.gettempdir()
    AUDIO_FOLDER = 'static/audio'
    SEPARATED_FOLDER = 'static/separated'
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mid', 'midi'}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024

    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
        os.makedirs(app.config['SEPARATED_FOLDER'], exist_ok=True) 