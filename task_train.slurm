#!/bin/bash
#SBATCH --job-name=compute channel stacks  # Job name
#SBATCH --nodes=1                       # Run on a single node 
#SBATCH --ntasks=1                       # Number of task (1 per GPU)
#SBATCH --gres=gpu:1                    # Request 1 GPUs
#SBATCH --partition=ASL-gpu             # Partition name for CPU jobs
#SBATCH --cpus-per-task=16              # Number if CPU cores per task
#SBATCH --mem=64G                      # Memory per node
#SBATCH --time=02:00:00                 # Time limit hrs:min:sec
#SBATCH --output=slurm-%j.out           # Standard output and error log
#SBATCH --error=%x_%j.err               # Standard error file

# Activate the virtual enviroment
source venv_mlproject/bin/activate

# Run the Python script
python scripts/train_model.py