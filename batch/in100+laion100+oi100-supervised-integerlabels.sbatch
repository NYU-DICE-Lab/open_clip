#!/bin/bash -x

#SBATCH --output=in100-laion100-oi100-sup-int-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=in100-laion100-oi100-sup-int
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
    /bin/bash "/scratch/bf996/open_clip/src/script/run-singularity.bash" \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python -u /scratch/bf996/open_clip/src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/in100_laion100_oi100_714406.csv" --csv-img-key path --csv-caption-key idx --csv-separator "," --integer-labels --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 4 --caption-subset=True --sim-clr-trans=True --precision=fp32 --warmup 2000 --batch-size=64 --norm_gradient_clip=5e4 --epochs=256 --workers=8 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/in100+oi100+laion100-supervised-integerlabels-ep1-168/checkpoints/epoch_168.pt" --local-loss --gather-with-grad'