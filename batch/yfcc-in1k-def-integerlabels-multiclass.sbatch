#!/bin/bash -x

#SBATCH --output=yfcc_in1kdef_intlabels_mc_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --job-name=yfcc_in1kdef_intlabels_mc
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_default_mc_3965496.csv" --csv-separator "," --integer-labels --multiclass  --csv-caption-key="in1k_subset_mc" --save-frequency 1 --warmup 2000 --batch-size=128 --precision=fp32 --epochs=64 --workers=8 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/yfcc-in1k-default-integerlabels-multiclass-ep1-62alt/checkpoints/epoch_62.pt" --local-loss --gather-with-grad'