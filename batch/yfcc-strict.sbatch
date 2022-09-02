#!/bin/bash -x

#SBATCH --output=yfcc_strict_new_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=yfcc_in1k_strict_new
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --dependency=afterany:22234381

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_strict_new_2272309.csv" --csv-separator "," --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --zeroshot-frequency=16 --save-frequency 1 --warmup 2000 --batch-size=256 --epochs=64 --workers=8 --model=RN50 --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-in1k-strict-new-ep1-32/checkpoints/epoch_31.pt" --local-loss --gather-with-grad'