#!/bin/bash -x

#SBATCH --output=laion400m-intlabels-strict-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=laion400m-intlabels-strict
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
    /bin/bash src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --report-to wandb --train-data "/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar" --train-num-samples 60000000 --dataset-type webdataset --integer-labels --strict=True --ds-filter="imagenet_classnames" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=2 --save-frequency 2 --warmup 2000 --batch-size=128 --epochs=32 --workers=8 --model=RN50-in1k --gather-with-grad --local-loss'