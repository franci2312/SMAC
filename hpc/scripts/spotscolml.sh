#!/bin/bash
#SBATCH -Jspotcml
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --array=0-29
#SBATCH -t 48:00:00 
#SBATCH --output=logs/spots/L-%A-%a.out
#SBATCH --error=logs/spots/L-%A-%a.err
#SBATCH -A your-account
#SBATCH -q embers    
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email

cd $SLURM_SUBMIT_DIR

module load anaconda3/2022.05.0.1
conda activate meshenv

mkdir -p logs/spots

# Configuration
METHOD="isomap" # Options: isomap, lle
TASK_ID=${SLURM_ARRAY_TASK_ID}
NEIG=3
INITLOC=0 # Change initloc to run the four scenarios of combined defects (0, 1, 2, 3)

FOLDER_NAME='bunnyssm'
BASE_DIR="/path/to/your/scratch/${FOLDER_NAME}/arlssm"
INPUT_FOLDER_CAD="${BASE_DIR}"
OUTPUT_FOLDER="${BASE_DIR}/output/${METHOD}/cspot$((INITLOC+1))/run$((TASK_ID+1))"

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

python spotscolml.py \
    --outputFolder "$OUTPUT_FOLDER" \
    --inputFolderCAD "$INPUT_FOLDER_CAD" \
    --neig "$NEIG" \
    --method "$METHOD" \
    --n_jobs 8 \
    --id $TASK_ID \
    --nbatches 7 \
    --size 1000 \
    --initloc $INITLOC \
    --date 1116 \
    --no 0

echo "Completed processing dataset for task idx: $TASK_ID"
