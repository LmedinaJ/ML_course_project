#!/bin/bash
#SBATHC --job-name=compute channel stacks  # Job name
#SBATHC --nodes=1                          # Run on a single node 
#SBATHC --ntask=1                          # Number of task (1 per GPU)
#SBATHC --gres=gpu:0                       # Request 1 GPUs
#SBATCH --partition=ASL-cpu                # Partition name for CPU jobs
#SBATHC --cpus-per-task=16                 # Number if CPU cores per task
#SBATHC --mem=128G                         # Memory per node
#SBATHC --time=02:00:00                    # Time limit hrs:min:sex
#SBATHC --output=slurm-%j.out              # Standard output and error log
#SBATCH --error=%x_%j.err                  # Standard error file

# Activate the virtual enviroment
source venv_mlproject/bin/activate

# Run the Python script
python scripts/stack_channels.py