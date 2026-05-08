#!/bin/bash
#SBATCH -JICML
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --array=0-29
#SBATCH -t 48:00:00 
#SBATCH --output=logs/icml/L-%A-%a.out
#SBATCH --error=logs/icml/L-%A-%a.err
#SBATCH -A your-account
#SBATCH -q embers    
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential

module load anaconda3/2022.05.0.1
conda activate meshenv

mkdir -p logs/icml

# Configuration
METHOD="isomap" # Options: isomap, lle  
TASK_ID=${SLURM_ARRAY_TASK_ID}
NEIG=3

FOLDER_NAME='bunnyssm' 
BASE_DIR="/path/to/your/scratch/${FOLDER_NAME}/arlssm"
INPUT_FOLDER_CAD="${BASE_DIR}"
OUTPUT_FOLDER="${BASE_DIR}/output/${METHOD}/ic/run$((TASK_ID+1))"

# Verify 
echo "=== Environment Check ==="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
echo "========================"

echo "Task ID: $TASK_ID"
echo "NEIG=$NEIG"
echo "INPUT_FOLDER_CAD=$INPUT_FOLDER_CAD"
echo "OUTPUT_FOLDER=$OUTPUT_FOLDER"

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER_CAD" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER_CAD"
    exit 1
fi

python icml.py \
    --outputFolder "$OUTPUT_FOLDER" \
    --inputFolderCAD "$INPUT_FOLDER_CAD" \
    --neig "$NEIG" \
    --method "$METHOD" \
    --n_jobs 8 \
    --id $TASK_ID \
    --nbatches 10 \
    --size 1000 \
    --date 1116 \
    --no 0

echo "Completed processing dataset for task idx: $TASK_ID"
