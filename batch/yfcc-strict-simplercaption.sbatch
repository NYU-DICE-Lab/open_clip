#!/bin/bash -x

#SBATCH --output=yfcc_strict_simplercaption_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:mi50:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=yfcc_strict_simplercaption
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --dependency=afterany:22234381

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_strict.csv" --csv-separator "," --csv-caption-key simple_title --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=256 --epochs=128 --workers=8 --model=RN50 --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-in1k-strict-simplercaptions-ep1-46/checkpoints/epoch_46.pt" --local-loss --gather-with-grad'