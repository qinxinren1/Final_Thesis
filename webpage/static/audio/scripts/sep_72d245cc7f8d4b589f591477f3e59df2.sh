#!/bin/bash
#SBATCH --job-name=sep_72d245cc7f8d4b589f591477f3e59df2
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=static/audio/scripts/sep_72d245cc7f8d4b589f591477f3e59df2.out

module load 2022r2
module load cuda/11.6
module load python/3.9.12

source $HOME/thesis/venv/bin/activate

python3 -m demucs.separate -n demucs_6s --two-stems=vocals,other -o "static/separated" "static/audio/72d245cc7f8d4b589f591477f3e59df2.mp3"
