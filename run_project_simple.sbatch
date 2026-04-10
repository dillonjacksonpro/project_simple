#!/bin/bash
#SBATCH --job-name=project_simple
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=rss-class
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --time=00:10:00

set -euo pipefail

# Hardcoded dataset directory for this job.
# Update this path to your actual input directory before submitting.
INPUT_DIR="/cluster/pixstor/class/cs4080/data/cc"
NUM_NODES=16

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: INPUT_DIR does not exist or is not a directory: $INPUT_DIR"
    exit 1
fi

# Build in the submission directory unless a different executable path is provided.
EXECUTABLE="${EXECUTABLE:-$SLURM_SUBMIT_DIR/project_simple}"
cd "$SLURM_SUBMIT_DIR"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Executable not found at $EXECUTABLE; building with make..."
    make
fi

# Use one OpenMP thread per allocated CPU.
OMP_NUM_THREADS=1

# Launch step with a clean exported environment and scrub conflicting mem vars
# from the srun process itself.
env \
    -u SLURM_MEM_PER_CPU \
    -u SLURM_MEM_PER_GPU \
    -u SLURM_MEM_PER_NODE \
    srun --cpus-per-task=1 --export=NONE,OMP_NUM_THREADS="$OMP_NUM_THREADS" "$EXECUTABLE" "$INPUT_DIR" "$NUM_NODES"
