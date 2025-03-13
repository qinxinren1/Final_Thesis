# open an interactive bash shell on a GPU-enabled node using SLURM's srun command

srun --partition=gpu --ntasks=1 --gpus-per-task=1 --cpus-per-task=1 --mem-per-cpu=1G --time=00:30:00 --pty bash

# check if the gpu is loaded

nvidia-smi

python -c "import torch; print(torch.cuda.is_available())"
