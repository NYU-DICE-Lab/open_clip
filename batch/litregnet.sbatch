#!/bin/bash -x

#SBATCH --output=litregnet_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_regnetx_320_laion400m_4node_b384
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu
#SBATCH --dependency=afterany:22042036
#SBATCH --dependency=afterany:22044929

module purge;

#debug flags
echo $SLURM_JOB_NAME

# COMMENTS
# resume point 2022_06_17-23_06_30 was trained on 350m LAION samples for the first 2 epochs

#env vars
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --save-frequency 1 --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/cc12m/{00000..01243}.tar" --train-num-samples 10968539 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --zeroshot-frequency=1 --warmup 2000 --batch-size=256 --wd=0.1 --epochs=4 --workers=4 --model=timm-regnetx_320 --pretrained-image --lock-image --local-loss --gather-with-grad'

