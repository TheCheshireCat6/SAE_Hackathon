# Run with vanilla SAE, configuration index 2
sbatch --export=ARCH=vanilla,IDX=2 slurm_sae.sh

# Run with topk SAE, configuration index 1
sbatch --export=ARCH=topk,IDX=1 slurm_sae.sh

# Run with jumprelu SAE, configuration index 0
sbatch --export=ARCH=jumprelu,IDX=0 slurm_sae.sh

# Run with batch_topk SAE, configuration index 1
sbatch --export=ARCH=batch_topk,IDX=1 slurm_sae.sh
