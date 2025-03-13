#!/bin/bash
#SBATCH --job-name=demucs_gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          
#SBATCH --gpus-per-task=1         
#SBATCH --mem-per-cpu=1G                  
#SBATCH --time=01:00:00
#SBATCH --output=demucs-%j.out

# Load modules and activate environment
module purge
module load cuda/11.7
module load conda
conda activate musicanalysis

# Run the script
python separate_tracks.py