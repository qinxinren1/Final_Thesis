#!/bin/bash
#SBATCH --job-name=demucs_separation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --output=demucs_%j.log

# Load necessary modules for DelftBlue
module purge
module load cuda/11.7
module load conda

# Activate the musicanalysis conda environment
source activate musicanalysis

# Input and output paths are passed as arguments
INPUT_FILE=$1
OUTPUT_DIR=$2
JOB_ID=$3

echo "Starting Demucs separation for file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Job ID: $JOB_ID"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run Demucs with GPU acceleration
# Using htdemucs model which separates into drums, bass, vocals, and other
python -m demucs.separate -n htdemucs $INPUT_FILE -o $OUTPUT_DIR

# Create a status file to indicate completion
echo "completed" > "${OUTPUT_DIR}/${JOB_ID}_status.txt"

echo "Separation completed. Results saved to: $OUTPUT_DIR" 