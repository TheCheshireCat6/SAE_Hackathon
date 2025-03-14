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
# 2. Configuration file name
# Usage: sbatch --export=ARCH=vanilla,CONFIG_FILE=test_config.yaml slurm_sae.sh

eval "$(conda shell.bash hook)"
conda activate architecture_search_env

# Default values if not provided through environment variables
ARCH=${ARCH:-topk}  # Default to topk
CONFIG_FILE=${CONFIG_FILE:-test_config.yaml}  # Default to test_config.yaml

# Print the configuration being used
echo "Running with architecture: $ARCH, config file: $CONFIG_FILE"

# Run the training script with the specific configuration
python Train_Enformer_SAE.py \
    --config configs/$CONFIG_FILE \
    --override-config configs/$ARCH.yaml

echo "Training completed"