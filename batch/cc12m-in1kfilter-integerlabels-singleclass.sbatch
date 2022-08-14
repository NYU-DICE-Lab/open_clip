#!/bin/bash -x

#SBATCH --output=cc12m-intlabels-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=384GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --job-name=cc12m-intlabels
#SBATCH --dependency=afterany:22820616
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
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data "/vast/work/public/ml-datasets/cc12m/{00000..01243}.tar" --train-num-samples 10968539 --dataset-type webdataset --integer-labels --ds-filter="imagenet_classnames" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/cc12m-RN50-integerlabels-ep1-22/checkpoints/epoch_21.pt" --gather-with-grad --local-loss'