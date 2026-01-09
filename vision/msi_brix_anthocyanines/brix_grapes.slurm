#!/bin/bash
# Example of running python script in a batch mode
#SBATCH --mail-user=name@domain.com
#SBATCH --mail-type=end,fail
#SBATCH --output=slurmop/brixgrapes128msi/all-out-brixindex-3dm128msi-StandardScaler-multigpuMNv2T1H4B4-run-13
#SBATCH --error=slurmop/brixgrapes128msi/err
#SBATCH --job-name="brix_msi"

eval "$(conda shell.bash hook)"
source activate tf2_15_1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
echo "TDM-MSI BrixIndex image regression slurm job started !"
echo $LD_LIBRARY_PATH
echo $XLA_FLAGS
srun python src/multigpu_brix_msi_grapes.py
echo "TDM-MSI BrixIndex image regression slurm job finished !"
