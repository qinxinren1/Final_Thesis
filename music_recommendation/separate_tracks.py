#!/usr/bin/env python3

import subprocess
import os

def separate_audio(input_file, output_dir='output'):
    """
    Separates audio tracks using Demucs with GPU acceleration.
    """
    # Ensure CUDA compatibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Demucs command with GPU flags
    command = [
        "demucs",
        "--device", "cuda",  # Force GPU usage
        "--float32",  # Better precision for GPU
        input_file,
        "-o", output_dir
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Tracks saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Demucs not found! Activate your Conda environment first.")

if __name__ == "__main__":
    # Use absolute paths for HPC compatibility
    input_audio = "input/EricClaptonUnplugged.mp3"  # Full path to file
    separate_audio(input_audio)