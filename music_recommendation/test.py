#!/usr/bin/env python3

import subprocess
import os

def separate_audio(input_file, output_dir='output'):
    """
    Separates the audio tracks of a given song using Demucs on GPU.

    Parameters:
    - input_file (str): Path to the input audio file.
    - output_dir (str): Directory where the separated tracks will be saved.
    
    This script uses the htdemucs_6s model and runs on the GPU.
    The separated stems will be saved under the specified output directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construct the Demucs command.
    # "-d cuda" tells Demucs to use the GPU.
    # "-n htdemucs_6s" selects the experimental 6-sources model.
    command = [
        "demucs",
        "-d", "cpu",
        "-n", "htdemucs_6s",
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
    # Replace with the path to your actual audio file.
    input_audio = "input/audio_07.mp3"
    separate_audio(input_audio)
