# OpenCLIP Batch Jobs

This brief readme gives an overview on how to run training jobs using OpenCLIP in NYU's HPC environment.

## Clone the Repo

Navigate to your working directory, eg, scratch/<USERNAME>

git clone https://github.com/NYU-DICE-Lab/open_clip

## Overlays

You will need to generate ext3 images use as overlays. Please follow the procedure described by [HPC](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda).

For NVIDIA CUDA, use this version of PyTorch in your image: pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

If you do not name your image openclip_env_cuda.ext3, you will need to modify run-singularity.bash accordingly.

For AMD ROCM, use this version of PyTorch in your image: pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2

If you do not name your image openclip_env_rocm.ext3, you will need to modify run-singularity-rocm.bash accordingly.

and don't forget:

```
pip install -r requirements-training.txt
```

## Basic sbatch template

You now have everything you need to run a job.

You can find example templates in the batch directory. These can be modified according to the particular needs of your run.

You will need to sync this repo to your scratch directory prior to first run. All batch scripts expect to be run from the open_clip directory.

Please be sure to replace any PATH variables as appropriate for your particular environment.

## Interactive, single-node jobs

For interactive srun jobs intended to be run multi-GPU, single-node, the syntax is considerably different than it is for SLURM.

the syntax for SLURM and torchrun is pretty different, so it won't work to try to reuse the SLURM commands. Instead:

1. Set your WORLD_SIZE, OMP_NUM_THREADS, MASTER_ADDR and MASTER_PORT environment variables (see the batch scripts in our repo for examples). Then, export SLURM_NTASKS=$WORLD_SIZE.

2. Launch a Singularity instance with the appropriate datasets mounted and available, navigate to your open_clip directory
3. Run the following command (modify nproc_per_node as needed)

```
torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --nnodes=1 --nproc_per_node=4 -m training.main *ARGS
```

If you run into trouble with the master address or ports, try re-running the command and see whether you get a different port the next time, or try fixing the port to a particular number (try the 50010 -> 50100 range).

This can be run from an interactive session in Greene using srun

```
srun --nodes=1 --cpus-per-task=8 --mem=128GB --gres=gpu:4 --time=47:59:00 --pty /bin/bash
```

* Run in tmux to avoid losing your job if your internet disconnects
* You may need to reset your MASTER_PORT in-between jobs if you cancel jobs in the middle or if they fail

## GPUs

### AMD

AMD Notes:

16GB VRAM is available on the AMD cards.

16 GPUs can be accessed using 2 nodes, 8 GPUs per node.

```bash
#SLURM
#SBATCH --gres=gpu:mi50:*

#SRUN
/bin/bash src/script/run-singularity-rocm.bash \
```

### NVIDIA

VRAM may be anywhere from 16GB to 40GB, depending on which cards you request or are assigned.

16 GPUs can be accessed using 4 nodes, 4 GPUs per node.

```bash
#SLURM
#SBATCH --gres=gpu:*

#SRUN
/bin/bash src/script/run-singularity.bash \
```

## Datasets (Train)

### CC3M

```bash
# SINGULARITY
--overlay /vast/work/public/ml-datasets/conceptual-captions/cc_data.sqf:ro

#OPENCLIP
--train-data="~/Train_GCC-training_output.csv"      
--val-data="~/Validation_GCC-1.1.0-Validation_output.csv"   
--csv-img-key filepath     
--csv-caption-key title
```

### CC12M

```bash
# OPENCLIP
--dataset-type webdataset \
--train-data "/vast/work/public/ml-datasets/cc12m/{00000..01243}.tar" \
--train-num-samples 10968539 \
```

### LAION400M

```bash
# OPENCLIP
## Feel free to adjust train-num-samples if you want to train on a subset of LAION
--dataset-type webdataset \
--train-data "/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar" \
--train-num-samples 400000000 \
```

### YFCC15M

```bash
# SLURM
#SBATCH --mem=128GB

# run-singularity.bash
$(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \

# OPENCLIP
--train-data="/vast/work/public/ml-datasets/yfcc15m/yfcc-small-metadata.csv" \
--csv-separator "," \
```

## Datasets (Validation)

### ImageNet Validation

You will need to request access from NYU HPC before you can use ImageNet

```bash
# SINGULARITY
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \

# OPENCLIP
--imagenet-val "/imagenet/val/" \
```

### Imagenet V2 Validation

ImageNet V2 should download and install automatically the first time your script calls it. Be sure to provide it with a path with write access.

```bash
--imagenet-v2 "INSTALL_PATH" \
```

## Arguments to OpenCLIP

The easiest way to get a sense of the arguments OpenCLIP accepts is to read the arg parser.

```
src/training/params.py
```