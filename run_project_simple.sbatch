#!/bin/bash
#SBATCH --job-name=project_simple
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=rss-class
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

set -euo pipefail

# Optional overrides when submitting:
#   sbatch --partition=compute --export=ALL,INPUT_DIR=/path/to/data run_project_simple.sbatch
INPUT_DIR="${INPUT_DIR:-}"
NUM_NODES=4

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: INPUT_DIR is not set."
    echo "Submit with: sbatch --export=ALL,INPUT_DIR=/path/to/data run_project_simple.sbatch"
    exit 1
fi

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

# Keep OpenMP deterministic by default; users can override OMP_NUM_THREADS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

srun "$EXECUTABLE" "$INPUT_DIR" "$NUM_NODES"
