#!/bin/bash -x

#SBATCH --output=in100-yfccmulti-laion100-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --job-name=in100-yfccmulti-laion100-titletagdescr
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
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python -u /scratch/bf996/open_clip/src/training/main.py --report-to wandb --train-data="/scratch/bf996/open_clip/yfcc-subsets/in100_yfcc_multi_laion100_1077537.csv" --csv-img-key path --csv-caption-key caption --csv-separator "," --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 4 --caption-subset=True --sim-clr-trans=True --warmup 2000 --batch-size=128 --epochs=256 --workers=4 --model=RN50 --resume "/scratch/bf996/open_clip/logs/in100+yfcc100+laion100-titletagdescr-ep1-156/checkpoints/epoch_156.pt" --local-loss --gather-with-grad'