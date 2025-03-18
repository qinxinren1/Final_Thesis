#!/usr/bin/env python3

import subprocess
import os
import argparse
import time
import sys

def separate_audio(input_file, output_dir='output', model='htdemucs_6s', segment=5, overlap=0.25, shifts=2, 
                  two_stems=None, mp3=False, mp3_bitrate=320, float32=True, int24=False, clip_mode='rescale'):
    """
    Separates audio tracks using Demucs with GPU acceleration.
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Directory to save the separated tracks
        model (str): Demucs model to use (htdemucs_6s for 6-stem separation)
        segment (int): Length of each segment in seconds (must be integer, recommended 5-6 for transformer models)
        overlap (float): Overlap between segments (0 to 1)
        shifts (int): Number of random shifts for equivariant stabilization (1-5)
        two_stems (str): If not None, separate into two stems only (vocals/instrumental)
        mp3 (bool): Save output as MP3 instead of WAV
        mp3_bitrate (int): Bitrate for MP3 encoding
        float32 (bool): Export in 32-bit float WAV
        int24 (bool): Export in 24-bit int WAV
        clip_mode (str): Strategy for avoiding clipping ('rescale' or 'clamp')
    """
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting audio separation process")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Input file: {input_file}")
    
    # Ensure CUDA compatibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Reduce memory usage
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Output directory created: {output_dir}")
    
    # Ensure segment is an integer
    segment = int(segment)
    
    # Demucs command with advanced options
    command = [
        "demucs",
        "--device", "cuda",  # Force GPU usage
        "-n", model,  # Use the specified model
        "--segment", str(segment),  # Split audio into segments to reduce memory
        "--overlap", str(overlap),  # Overlap between segments
        "--shifts", str(shifts),  # Random shifts for equivariant stabilization
        "--clip-mode", clip_mode,  # How to handle clipping
        "--verbose",  # More detailed output
    ]
    
    # Add optional parameters
    if float32:
        command.append("--float32")
    if int24:
        command.append("--int24")
    if mp3:
        command.append("--mp3")
        command.extend(["--mp3-bitrate", str(mp3_bitrate)])
    if two_stems:
        command.extend(["--two-stems", two_stems])
    
    # Add input file and output directory
    command.extend([input_file, "-o", output_dir])
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running command: {' '.join(command)}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated time: {estimate_processing_time(input_file, model, shifts)} minutes")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Separation started - this may take a while...")
    
    # Flush stdout to ensure progress messages are visible in real-time
    sys.stdout.flush()
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  universal_newlines=True, bufsize=1)
        
        # Print output in real-time with timestamps
        for line in process.stdout:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {line.strip()}")
            sys.stdout.flush()
        
        # Wait for process to complete
        process.wait()
        
        # Check return code
        if process.returncode != 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: Demucs returned non-zero exit code: {process.returncode}")
            if process.returncode == -9:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] This is likely an out-of-memory error. Try reducing segment size or shifts.")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Recommended: Try with segment=3 and shifts=1")
            return
        
        output_path = f"{output_dir}/{model}/{os.path.basename(input_file).split('.')[0]}"
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Tracks saved to: {output_path}")
        
        # Print information about the stems that were created
        if model == "htdemucs_6s":
            print("\nThe 6-stem model creates the following stems:")
            print("- vocals.wav: Vocal track")
            print("- drums.wav: Drum track")
            print("- bass.wav: Bass track")
            print("- other.wav: Other instruments")
            print("- guitar.wav: Guitar track")
            print("- piano.wav: Piano track (note: may have quality issues)")
        elif two_stems:
            print(f"\nTwo-stem separation created:")
            print(f"- {two_stems}.wav: The selected stem")
            print(f"- no_{two_stems}.wav: Everything else")
        else:
            print("\nThe 4-stem model creates the following stems:")
            print("- vocals.wav: Vocal track")
            print("- drums.wav: Drum track")
            print("- bass.wav: Bass track")
            print("- other.wav: Other instruments")
        
        # Check if all expected files were created
        expected_stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
        if model == "htdemucs_6s":
            expected_stems.extend(["guitar.wav", "piano.wav"])
        elif two_stems:
            expected_stems = [f"{two_stems}.wav", f"no_{two_stems}.wav"]
        
        missing_stems = []
        for stem in expected_stems:
            stem_path = os.path.join(output_path, stem)
            if not os.path.exists(stem_path):
                missing_stems.append(stem)
        
        if missing_stems:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Some expected stems are missing: {', '.join(missing_stems)}")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All expected stems were created successfully")
        
        # Print total processing time
        elapsed_time = time.time() - start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}")
    except FileNotFoundError:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Demucs not found! Make sure it's installed with: pip install demucs")

def estimate_processing_time(input_file, model, shifts):
    """
    Provides a rough estimate of processing time based on file size and model.
    This is a very rough estimate and actual times will vary.
    """
    try:
        # Get file size in MB
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        
        # Base processing factor (MB per minute) - empirically determined
        # These are rough estimates and will vary by hardware
        if model == "htdemucs_6s":
            base_factor = 5  # MB per minute for 6-stem model
        elif model == "htdemucs_ft":
            base_factor = 7  # MB per minute for fine-tuned model
        else:
            base_factor = 10  # MB per minute for other models
        
        # Adjust for shifts (each shift roughly doubles processing time)
        shift_factor = shifts / 2
        
        # Calculate estimated minutes
        estimated_minutes = (file_size_mb / base_factor) * shift_factor
        
        return max(1, round(estimated_minutes))
    except Exception as e:
        print(f"Error estimating processing time: {e}")
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Separate audio tracks using Demucs")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-m", "--model", default="htdemucs_6s", 
                        choices=["htdemucs_6s", "htdemucs_ft", "htdemucs", "hdemucs_mmi", "mdx", "mdx_extra"],
                        help="Model to use for separation (htdemucs_6s for 6 stems)")
    parser.add_argument("-s", "--segment", type=int, default=5, 
                        help="Length of each segment in seconds (must be an integer, recommended 5-6 for transformer models)")
    parser.add_argument("--overlap", type=float, default=0.25, 
                        help="Overlap between segments (0 to 1)")
    parser.add_argument("--shifts", type=int, default=2, 
                        help="Number of random shifts for equivariant stabilization (1-5)")
    parser.add_argument("--two-stems", choices=["vocals", "drums", "bass", "other", "guitar", "piano"], 
                        help="Separate into two stems only (e.g., vocals/instrumental)")
    parser.add_argument("--mp3", action="store_true", help="Save output as MP3 instead of WAV")
    parser.add_argument("--mp3-bitrate", type=int, default=320, help="Bitrate for MP3 encoding")
    parser.add_argument("--float32", action="store_true", help="Export in 32-bit float WAV")
    parser.add_argument("--int24", action="store_true", help="Export in 24-bit int WAV")
    parser.add_argument("--clip-mode", default="rescale", choices=["rescale", "clamp"], 
                        help="Strategy for avoiding clipping")
    
    args = parser.parse_args()
    
    separate_audio(
        args.input_file, 
        args.output, 
        args.model, 
        args.segment, 
        args.overlap, 
        args.shifts,
        args.two_stems, 
        args.mp3, 
        args.mp3_bitrate, 
        args.float32, 
        args.int24, 
        args.clip_mode
    )

if __name__ == "__main__":
    main()