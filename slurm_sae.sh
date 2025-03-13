#!/bin/bash
#SBATCH --job-name=pytorch_hackathon_enformer_bteam
#SBATCH --output=pytorch_hackathon_enformer_bteam.out
#SBATCH --error=pytorch_hackathon_enformer_bteam.err
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --qos=hackathon
#SBATCH --account=hackathon_202503
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=lemanczyk@cshl.edu
#SBATCH --partition=hackathonq

# This script expects two arguments:
# 1. SAE architecture (vanilla, topk, jumprelu, batch_topk)
# 2. Configuration index
# Usage: sbatch --export=ARCH=vanilla,IDX=0 slurm_sae.sh

eval "$(conda shell.bash hook)"
conda activate architecture_search_env

# Default values if not provided through environment variables
ARCH=${ARCH:-topk}  # Default to topk
IDX=${IDX:-0}       # Default to index 0

# Print the configuration being used
echo "Running with architecture: $ARCH, index: $IDX"

# Run the training script with the specific configuration
python Train_Enformer_SAE.py \
    --base-config configs/base_enformer_config.yaml \
    --sae-type $ARCH \
    --override-config configs/$ARCH/$IDX.yaml

echo "Training completed"
