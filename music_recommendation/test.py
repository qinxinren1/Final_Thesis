#!/usr/bin/env python3

import subprocess
import os

def separate_audio(input_file, output_dir='output'):
    """
    Separates the audio tracks of a given song using Demucs.

    Parameters:
    - input_file (str): Path to the input audio file.
    - output_dir (str): Directory where the separated tracks will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construct the Demucs command.
    # The "--device cuda" flag ensures that Demucs uses the GPU.
    command = [
        "demucs",
        input_file,
        "-o", output_dir
    ]
    
    try:
        # Run the command and wait for it to complete.
        subprocess.run(command, check=True)
        print(f"Separation complete. Files saved in '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print("Error during separation:", e)

if __name__ == "__main__":
    # Replace 'your_song.mp3' with the path to your audio file.
    input_audio = "input/audio_14.mp3"
    separate_audio(input_audio)
