#!/bin/bash -x

#SBATCH --output=yfcc_lit_RN50_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=yfcc_lit_rn50
#SBATCH --dependency=afterany:22685019
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

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
    /bin/bash src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data="/scratch/bf996/datasets/yfcc15m/yfcc-small-metadata.csv" --csv-separator "," --resume "/scratch/bf996/open_clip/logs/yfcc-RN50timm-lit-ep8-31/checkpoints/epoch_31.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=256 --epochs=32 --workers=8 --model=timm-resnet50 --pretrained-image --lock-image --local-loss --gather-with-grad'