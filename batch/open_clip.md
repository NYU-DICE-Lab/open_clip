### Environment

#### Local

First navigate to the open_clip directory. Then run --

```bash
conda activate open_clip; export PYTHONPATH="$PYTHONPATH:$PWD/src"; 
#sometimes you also need
pip install -r requirements-training.txt; 
```

#### SLURM GPU

```bash
cd /scratch/bf996/open_clip

singularity exec --nv \
  $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/openclip_env_cuda.ext3:ro \
  --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-sketch.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash
```

### SLURM CPU

```bash
cd /scratch/bf996/open_clip

singularity exec \
  $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/openclip_env_cuda.ext3:ro \
  --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-sketch.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash
```

### ROCM

```bash
singularity \
    exec --rocm \
    --bind $tmp:$HOME/.config/miopen \
  $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/openclip_env_rocm_25.ext3:rw \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-sketch.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/rocm5.2.0-ubuntu20.04.4.sif \
  /bin/bash
```

### Other Rocm
/scratch/work/public/singularity/rocm5.1.1-ubuntu20.04.4.sif
/scratch/work/public/singularity/rocm5.2.0-ubuntu20.04.4.sif
/scratch/work/public/hudson/images/rocm4.5.2-ubuntu20.04.3.sif

### GPU FLAGS

```bash
#NO GPU
source /ext3/env.sh; export PYTHONPATH="$PYTHONPATH:/scratch/bf996/open_clip/src"; unset SLURM_NTASKS;

#ONE GPU
source /ext3/env.sh; export PYTHONPATH="$PYTHONPATH:/scratch/bf996/open_clip/src";

#MULTI-GPU
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; export MASTER_PORT=$(shuf -i 10000-65500 -n 1); export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"; source /ext3/env.sh; export PYTHONPATH="$PYTHONPATH:/scratch/bf996/open_clip/src";

#CLEAN OPE NCLIP
source /ext3/env.sh; export PYTHONPATH="$PYTHONPATH:/scratch/bf996/clean_open_clip/src";
```

### DEBUG

```bash
#TORCH
export NCCL_P2P_LEVEL=NVL; export NCCL_BLOCKING_WAIT=1; export NCCL_DEBUG=INFO; export TORCH_CPP_LOG_LEVEL=INFO; export TORCH_DISTRIBUTED_DEBUG=INFO; export TORCH_SHOW_CPP_STACKTRACES=1; export NCCL_DEBUG_SUBSYS=ALL; export PYTHONFAULTHANDLER=1;

#ROCM
export HCC_SERIALIZE_KERNEL=0x3; export HCC_SERIALIZE_COPY=0x3; export HIP_TRACE_API=0x2; export MIOPEN_ENABLE_LOGGING_CMD=1;
```

### Commands

#### 1 Node, 1 GPU

python -u src/training/main.py

#### 1 Node, Multi GPU

torchrun --nproc_per_node 2 -m training.main *ARGS

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main *ARGS

### Flags

#### Cosmetic

--name "ViT-B-32-Vanilla-l50m-03"

--report-to wandb,tensorboard

#### Hyperparameters

--save-frequency 1

--batch-size=4096

--epochs=3  

--workers=8

--workers=$OMP_NUM_THREADS

--lr=1e-3     

--wd=0.1

--warmup=1000

--precision=fp32

--precision='amp'

#### LiT Tuning

--lock-image

#### Gradient Caching

--batch-size=2048  --gc=True  --gpumaxbatch=128

#### Declip

--model=timm-efficientnetv2_rw_s  --mlm=True

#### Eval

--zeroshot-frequency=1

--food "/" 

--air "/"

--stanfordcars "/"

--flowers "/"

--imagenet-val "/imagenet/val/"

--imagenet-a "/imagenet-a" 

--imagenet-r "/imagenet-r"

--imagenet-v2 "/scratch/bf996/datasets"

--imagenet-s "/imagenet-sketch"

--inat2021 "/scratch/bf996/datasets/"

--caption-subset=True

#### Dataset Type

--train-data="~/Train_GCC-training_output.csv"       --csv-img-key filepath     --csv-caption-key title

--train-data '/vast/work/public/ml-datasets/laion400m/{7500..22500}.tar' --train-num-samples 50000000 --dataset-type webdataset

--imagenet-train-data="/imagenet/train"

#### Dataset Modification

--ds-filter="imagenet_classnames"

--csv-cleaned=True

--csv-scrambled=True

--zeroshot-scramble=True

--ds-cipher=True

--simplecaptions=True

#### Model Selection

--model=RN50  --pretrained=openai

--resume "/scratch/bf996/open_clip/logs/yfcc_cars_01/checkpoints/epoch_32.pt"

--model=ViT-L-14 --pretrained=laion400m_e32

--model=ViT-B-32 --pretrained=laion2b_e16

--model=ViT-B-32 --pretrained=laion400m_e32

--model timm-vit_large_patch16_384_pretrained

--model timm-resnet50d

### pretrained model list

[('RN50', 'openai'),
 ('RN50', 'yfcc15m'),
 ('RN50', 'cc12m'),
 ('RN50-quickgelu', 'openai'),
 ('RN50-quickgelu', 'yfcc15m'),
 ('RN50-quickgelu', 'cc12m'),
 ('RN101', 'openai'),
 ('RN101', 'yfcc15m'),
 ('RN101-quickgelu', 'openai'),
 ('RN101-quickgelu', 'yfcc15m'),
 ('RN50x4', 'openai'),
 ('RN50x16', 'openai'),
 ('ViT-B-32', 'openai'),
 ('ViT-B-32', 'laion400m_e31'),
 ('ViT-B-32', 'laion400m_e32'),
 ('ViT-B-32', 'laion400m_avg'),
 ('ViT-B-32-quickgelu', 'openai'),
 ('ViT-B-32-quickgelu', 'laion400m_e31'),
 ('ViT-B-32-quickgelu', 'laion400m_e32'),
 ('ViT-B-32-quickgelu', 'laion400m_avg'),
 ('ViT-B-16', 'openai'),
 ('ViT-L-14', 'openai')]

### Example Commands

#### INFERENCE ON PRETRAINED MODEL CHECKPOINT

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --pretrained=openai --caption-subset=True; 

python src/training/main.py --batch-size=32 --workers=8 --imagenet-val "/imagenet/val/" --zeroshot-frequency=1  --model=RN50  --pretrained=openai

python src/training/main.py --batch-size=32 --workers=4 --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --pretrained=yfcc15m  --precision=fp32

python src/training/main.py --batch-size=32 --workers=4 --inat2021="/scratch/bf996/datasets/" --zeroshot-frequency=1  --model=RN50  --pretrained=yfcc15m 

python -u /scratch/bf996/open_clip/src/training/main.py --stanfordcars "/scratch/bf996/datasets/stanfordcars" --flowers "/scratch/bf996/datasets/flowers102" --zeroshot-frequency=1 --workers=16 --model=RN50 --pretrained=yfcc15m

python src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/cc12m/{00000..01243}.tar" --train-num-samples 10968539 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --ds-filter="imagenet_classnames" --zeroshot-frequency=4 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=8 --precision=fp32 --norm_gradient_clip=1e5 --model=RN50

python -u /scratch/bf996/open_clip/src/training/main.py --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1 --workers=8 --model=RN50 --pretrained=openai --no-ensembling=True --zeroshot-scramble=True

python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model=RN50 --pretrained=openai --caption-subset=True

python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/"  --imagenet-a "/imagenet-a"  --imagenet-r "/imagenet-r" --model=ViT-B-32 --pretrained=laion2b_e16

python -u /scratch/bf996/open_clip/src/training/main.py --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model=RN50  --pretrained=yfcc15m --no-ensembling=True; python -u /scratch/bf996/open_clip/src/training/main.py --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model=RN50  --pretrained=yfcc15m --no-ensembling=True --zeroshot-scramble=True

python -u /scratch/bf996/open_clip/src/training/main.py --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model=RN50 --pretrained=yfcc15m --caption-subset=True

python -u /scratch/bf996/open_clip/src/training/main.py --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion15m-vitb16-fromsimclr-ep1-5/checkpoints/epoch_5.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model="timm-vit_base_patch16_224_1k" --add-trunk=True

#### Resume VL from Pretrained SimCLR

python -u /scratch/bf996/open_clip/src/training/main.py --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model="timm-vit_base_patch16_224_1k" --pretrained-head="/scratch/bf996/open_clip/logs/laion15m-vit-b-16-simclr-ep7/checkpoints/epoch_7.pt" --norm_gradient_clip=1e5 --local-loss --gather-with-grad

#### Resume Integer Labels from Pretrained SIMCLR

python src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{31000..32500}.tar" --train-num-samples 15000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=32 --epochs=32 --workers=4 --model="timm-vit_base_patch16_224_1k" --integer-labels --ds-filter="imagenet_classnames" --pretrained-head="/scratch/bf996/open_clip/logs/laion15m-vit-b-16-simclr-ep7/checkpoints/epoch_7.pt"

#### MLFound Version

python -m training.main \
    --imagenet-val "/imagenet/val/" \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32

#### INFERENCE ON TRAINED MODEL CHECKPOINT

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/laion15m-RN50-proper-ep24-32/checkpoints/epoch_32.pt" --caption-subset=True

--imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --air "/" --stanfordcars "/" --food "/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion15m-lit-igresnext-ep1-2/checkpoints/epoch_2.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model=timm-igresnext32x48;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion400m-15mtrain-swin-lit-ul4-ep1-7/checkpoints/epoch_2.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model=timm-swin_base_patch4_window7_224;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-simplenounadj-ep24-28-redux/checkpoints/epoch_26.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model=RN50;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-filip-ep1-23/checkpoints/epoch_23.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model="resnet50" --filip=True;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion50m-swin-coca-ep3/checkpoints/epoch_3.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch"  --zeroshot-frequency=1 --model="coca"

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/regnet-ep4/checkpoints/lit_regnet_ep4.pt" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --air "/" --stanfordcars "/" --food "/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1 --model=timm-regnetx_320 --pretrained-image;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/effnet-ep6/checkpoints/epoch_6.pt" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --air "/" --stanfordcars "/" --food "/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1 --model=timm-tf_efficientnetv2_xl_in21ft1k --pretrained-image;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/yfcc-vit-b-32-declip-ep21-32/checkpoints/epoch_32.pt" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1 --model=vit_base_patch32_224 --mlm=True;

#### Shift Cipher Validation

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/2022_07_21-13_05_14-model_RN50-lr_0.0005-b_256-j_8-p_amp/checkpoints/epoch_4.pt" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --shift-cipher=3 --zeroshot-frequency=1 --model=RN50

#### LINEAR PROBE

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-r "/imagenet-r" --model="coat_tiny" --zeroshot-frequency=1 --linear-probe=True --image-size=224

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model="resnet50" --zeroshot-frequency=1 --linear-probe=True --caption-subset=True

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model="vit_huge_patch14_224_in21k" --zeroshot-frequency=1 --linear-probe=True --image-size=224

vit_huge_patch14_224_in21k

ig_resnext101_32x48d

tf_efficientnet_l2_ns

regnety_032

regnety_002

vgg19_bn

tf_efficientnetv2_xl_in21ft1k

swin_base_patch4_window7_224

regnetx_320

vit_base_patch32_224

vit_tiny_patch16_224

resnet50

resnet101

resnext101_64x4d

#### SINGLE NODE TRAINING

python -u /scratch/bf996/open_clip/src/training/main.py --train-data="/scratch/bf996/datasets/yfcc15m/yfcc-small-metadata.csv" --csv-separator "," --imagenet-val "/imagenet/val/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --simplecaptions=True --csv-cleaned=True --zeroshot-frequency=2 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=16 --workers=16 --debug --model=RN50

#### SINGLE NODE TRAINING with SIMCLR

python -u /scratch/bf996/open_clip/src/training/main.py --train-data="/scratch/bf996/datasets/yfcc15m/yfcc-small-metadata.csv" --csv-separator "," --sim-clr=True --save-frequency 1 --warmup 2000 --batch-size=32 --epochs=16 --workers=8 --model="vit_base_patch16_224" --use-bn-sync

#### Single Node Training with Imagenet-Captions-100

python -u /scratch/bf996/open_clip/src/training/main.py --train-data="/scratch/bf996/imagenet-captions/imagenet-captions-201-complete.csv" --csv-img-key path --csv-caption-key caption --csv-separator "," --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --caption-subset=True --lr=1e-3 --sim-clr-trans=True --warmup 2000 --batch-size=32 --epochs=128 --workers=8 --model=RN50

#### Single Node Training with Schema

python -u /scratch/bf996/open_clip/src/training/main.py --schema="/scratch/bf996/open_clip/schema/cc12m-l15m-32.txt" --imagenet-val "/imagenet/val/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --simplecaptions=True --csv-cleaned=True --zeroshot-frequency=2 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=16 --workers=8 --model=RN50

#### SimCLR Inference

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion15m-vit-b-16-simclr-ep3-6/checkpoints/epoch_6.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model="vit_base_patch16_224" --use-bn-sync --sim-clr=True

#### Single Node Training with Integer Multiclass labels

python -u /scratch/bf996/open_clip/src/training/main.py --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_mc_3965496.csv" --csv-separator "," --integer-labels --multiclass --csv-caption-key="in1k_subset_mc" --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=8 --model=RN50 --precision=fp32

python -u /scratch/bf996/open_clip/src/training/main.py --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --dataset-type webdataset --integer-labels --multiclass --ds-filter="imagenet_classnames" --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=8 --model=RN50 --precision=fp32

python src/training/main.py --dataset-type webdataset --train-data "/scratch/bf996/imagenet-captions/split/{0000..0141}.tar" --train-num-samples 1200000 --integer-labels --strict=True --ds-filter="imagenet_captions_classnames" --zeroshot-frequency=8 --save-frequency 1 --lr=1e-3 --warmup 500 --batch-size=256 --epochs=128 --workers=8 --model=RN50-in1k

#### Single Node Dry Run with Integer Labels

python -u /scratch/bf996/open_clip/src/training/main.py --train-data "/vast/work/public/ml-datasets/laion400m/{00000..00010}.tar" --train-num-samples 100000 --dataset-type webdataset --integer-labels --multiclass --ds-filter="imagenet_classnames" --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=1 --workers=1 --model=RN50 --dry-run=True --precision=fp32

#### Multi-Node Training with Integer labels

torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_single_3965496.csv" --csv-separator "," --integer-labels  --csv-caption-key="in1k_subset" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=RN50-in1k

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_mc_3965496.csv" --csv-separator "," --integer-labels --multiclass  --csv-caption-key="in1k_subset_mc" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=timm-vit_base_patch16_224_1k

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{11000..12500}.tar" --train-num-samples 15000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model="timm-vit_base_patch16_224_1k" --integer-labels --ds-filter="imagenet_classnames" --pretrained-head="/scratch/bf996/open_clip/logs/laion15m-vit-b-16-simclr-ep7/checkpoints/epoch_7.pt" --norm_gradient_clip=1e5 --local-loss --gather-with-grad

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model="vit_base_patch16_224" --resume "/scratch/bf996/open_clip/logs/laion15m-vit-b-16-simclr-ep3-6/checkpoints/epoch_6.pt" --use-bn-sync --sim-clr=True --local-loss --gather-with-grad

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --dataset-type webdataset --integer-labels --multiclass --ds-filter="imagenet_classnames" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 1 --warmup 2000 --norm_gradient_clip=1e5 --batch-size=128 --epochs=32 --workers=4 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/laion15m-in1k-integerlabels-multiclass-ep8-14/checkpoints/epoch_14.pt" --gather-with-grad --local-loss

torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --dataset-type webdataset --integer-labels --multiclass --ds-filter="imagenet_classnames" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 1 --warmup 2000 --batch-size=64 --epochs=32 --workers=8 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/laion15m-in1k-integerlabels-multiclass-ep8-14/checkpoints/epoch_14.pt" --gather-with-grad --local-loss --precision=fp32

##### Strict, with Pretrained Image

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_in1k_mc_3965496.csv" --csv-separator "," --integer-labels --multiclass  --csv-caption-key="in1k_subset_mc" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=16 --resume "/scratch/bf996/open_clip/logs/yfcc-vit-b-16-pretrained-integerlabels-strict-ep1-3/checkpoints/epoch_3.pt" --save-frequency 1 --seed 0 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=timm-vit_base_patch16_224_1k --pretrained-image --strict=True

#### Linear Probe on Int Labels

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-multiclass-ep1-62alt/checkpoints/epoch_62.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --model=RN50-in1k --zeroshot-frequency=1 --integer-labels

#### Metadata Captions

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --dataset-type webdataset --train-data "/scratch/bf996/imagenet-captions/split/{0000..0141}.tar" --train-num-samples 1200000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=256 --ds-filter="imagenet_classnames" --metacaptions="/scratch/bf996/open_clip/metadata/in1k_metadata.csv" --epochs=128 --workers=8 --model=RN50 --local-loss --gather-with-grad

#### LiT Tuning

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{10000..11500}.tar" --train-num-samples 15000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=256 --epochs=32 --workers=8 --model=timm-beit_large_patch16_224 --pretrained-image --lock-image --local-loss --gather-with-grad

#### Shift Cipher Experiments

python -u /scratch/bf996/open_clip/src/training/main.py --train-data="/scratch/bf996/open_clip/yfcc-subsets/yfcc_strict.csv" --csv-separator "," --imagenet-val "/imagenet/val/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --shift-cipher=3 --csv-cleaned=True --zeroshot-frequency=2 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=16 --workers=16 --model=RN50

python -u /scratch/bf996/open_clip/src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..00010}.tar" --train-num-samples 100000 --imagenet-val "/imagenet/val/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --shift-cipher=3 --csv-cleaned=True --zeroshot-frequency=2 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=16 --workers=8 --debug --model=RN50

#### vssl

python src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=8 --model=vit_large_patch16_224 --vssl=True --local-loss --gather-with-grad

#### filip

python src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..01500}.tar" --train-num-samples 15000000 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=8 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=8 --model=vit_large_patch16_224 --filip=True --local-loss --gather-with-grad

#### imagenet-tuning interleaved

python -u /scratch/bf996/open_clip/src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..00010}.tar" --imagenet-tune-freq=2 --imagenet-train="/imagenet/train" --train-num-samples 100000 --imagenet-val "/imagenet/val/" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r"  --zeroshot-frequency=2 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=16 --workers=4 --model=RN50-in1k

#### WDS TRAINING WITH FILTERING

python -u /scratch/bf996/open_clip/src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..10000}.tar" --train-num-samples 15000000 --imagenet-val "/imagenet/val/" --ds-filter="imagenet_classnames" --gc=True  --gpumaxbatch=128 --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=1024 --epochs=16 --workers=8 --model=ViT-B-32

#### WDS TRAINING WITHOUT FILTERING

python src/training/main.py --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..10000}.tar" --train-num-samples 500000 --imagenet-val "/imagenet/val/" --zeroshot-frequency=2 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=128 --epochs=16 --workers=4 --debug --model=RN50

#### DISTRIBUTED TRAINING

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --report-to wandb --train-data="/scratch/bf996/datasets/yfcc15m/yfcc-small-metadata.csv" --csv-separator "," --imagenet-val "/imagenet/val/" --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=384 --epochs=32 --workers=4 --model=ViT-B-32 --local-loss --gather-with-grad

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m training.main --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..00100}.tar" --train-num-samples 50000 --imagenet-val "/imagenet/val/" --ds-filter="imagenet_classnames" --csv-cleaned=True --zeroshot-frequency=4 --save-frequency 1 --seed 0 --warmup 2000 --batch-size=384 --epochs=32  --model=ViT-B-32 --local-loss --gather-with-grad

####

python src/training/main.py --train-data "/vast/work/public/ml-datasets/laion400m/{20000..30000}.tar" --train-num-samples 15000000 --dataset-type webdataset --integer-labels --multiclass --strict=True --ds-filter="imagenet_classnames" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=4 --save-frequency 1 --warmup 2000 --batch-size=128 --epochs=32 --workers=4 --model=RN50-in1k --resume "/scratch/bf996/open_clip/logs/laion100m-15mtrain-in1k-filter-RN50-ep4-20/checkpoints/epoch_19.pt"

### Inference Megacommand

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --pretrained=openai --caption-subset=True; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50 --pretrained=openai; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/laion15m-RN50-proper-ep24-32/checkpoints/epoch_32.pt" --caption-subset=True; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50 --resume "/scratch/bf996/open_clip/logs/laion15m-RN50-proper-ep24-32/checkpoints/epoch_32.pt"; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion15m-lit-igresnext-ep3/checkpoints/epoch_3.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model=timm-igresnext32x48; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion15m-lit-igresnext-ep1-2/checkpoints/epoch_2.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model=timm-igresnext32x48; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion400m-15mtrain-swin-lit-ul4-ep1-7/checkpoints/epoch_2.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model=timm-swin_base_patch4_window7_224; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --resume "/scratch/bf996/open_clip/logs/laion400m-15mtrain-swin-lit-ul4-ep1-7/checkpoints/epoch_4.pt" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --zeroshot-frequency=1 --model=timm-swin_base_patch4_window7_224;  python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-ep29-32/checkpoints/epoch_32.pt" --caption-subset=True; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-in1k-strict-new-ep1-56/checkpoints/epoch_56.pt" --integer-labels --caption-subset=True; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-in1k-strict-new-ep1-56/checkpoints/epoch_56.pt" --integer-labels; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-in1k-strict-new-ep1-56/checkpoints/epoch_32.pt" --integer-labels; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-in1k-strict-new-ep1-56/checkpoints/epoch_32.pt" --integer-labels --caption-subset=True;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50 --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-in1k-strict-new-ep33-64/checkpoints/epoch_64.pt"  --caption-subset=True;

python src/training/main.py --batch-size=32 --workers=8 --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/laion15m-RN50-proper-ep24-32/checkpoints/epoch_32.pt" --caption-subset=True --extended-metrics=True; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-ep29-32/checkpoints/epoch_32.pt" --caption-subset=True --extended-metrics=True;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/laion15m-in1k-integerlabels-multiclass-ep19-32/checkpoints/epoch_32.pt" --integer-labels --caption-subset=True; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/laion15m-in1k-integerlabels-multiclass-ep19-32/checkpoints/epoch_32.pt" --integer-labels; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-multiclass-ep1-62alt/checkpoints/epoch_62.pt" --integer-labels --caption-subset=True; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/yfcc-RN50-integerlabels-multiclass-ep1-62alt/checkpoints/epoch_62.pt" --integer-labels;

in100+yfcc100 supervised ep112
in100 supervised ep36

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/in100-supervised-ep1-128/checkpoints/epoch_36.pt" --caption-subset=True --extended-metrics=True;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/in100-titletagdescr-ep41-128/checkpoints/epoch_120.pt" --caption-subset=True --extended-metrics=True;

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/in100+yfcc100-titletagdescr-ep1-256/checkpoints/epoch_220.pt" --caption-subset=True --extended-metrics=True; python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50  --resume "/scratch/bf996/open_clip/logs/in100-bliptitle-ep1-128/checkpoints/epoch_112.pt" --caption-subset=True --extended-metrics=True; 

python src/training/main.py --batch-size=32 --workers=8 --report-to wandb --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/laion100m-15mtrain-in1k-filter-RN50-ep30-32/checkpoints/epoch_32.pt" --integer-labels --caption-subset=True;

python src/training/main.py --batch-size=32 --workers=8 --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --zeroshot-frequency=1  --model=RN50-in1k  --resume "/scratch/bf996/open_clip/logs/laion100m-15mtrain-in1k-filter-RN50-ep30-32/checkpoints/epoch_32.pt" --integer-labels --caption-subset=True --extended-metrics=True;