#!/bin/bash -x

#SBATCH --output=yfcc-not-in1k-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --job-name=yfcc-not-in1k
#SBATCH --dependency=afterany:22179172
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_not_in1k_10264405.csv" --csv-separator "," --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 4 --warmup 2000 --batch-size=256 --epochs=64 --workers=8 --model=RN50 --resume "/scratch/bf996/open_clip/logs/yfcc-not-in1k-ep1-19/checkpoints/epoch_16.pt" --local-loss --gather-with-grad'