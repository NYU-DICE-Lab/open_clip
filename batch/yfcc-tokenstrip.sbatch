#!/bin/bash -x

#SBATCH --output=yfcc-tokenstrip-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=yfcc-tokenstrip
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --dependency=afterany:22138171

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
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/datasets/yfcc15m/yfcc-small-metadata.csv" --token-scrambled=True --csv-separator "," --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --token-strip=True --caption-subset=True --zeroshot-frequency=4 --save-frequency 4 --warmup 2000 --batch-size=256 --epochs=32 --workers=8 --resume "/scratch/bf996/open_clip/logs/yfcc-tokenstrip-ep1-4/checkpoints/epoch_4.pt" --model=RN50 --local-loss --gather-with-grad'