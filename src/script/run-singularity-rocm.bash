#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

tmp=/tmp/$USER/$$
if [[ "$SLURM_TMPDIR" != "" ]]; then
    tmp="$SLURM_TMPDIR/miopen/$$"
fi
mkdir -p $tmp

singularity \
    exec --rocm \
    --bind $tmp:$HOME/.config/miopen \
  $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/openclip_env_rocm.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /scratch/bf996/datasets/fgvc-aircraft-2013b.sqf:ro \
  --overlay /scratch/bf996/datasets/flowers-102.sqf:ro \
  --overlay /scratch/bf996/datasets/stanford_cars.sqf:ro \
  --overlay /scratch/bf996/datasets/food-101.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-sketch.sqf:ro \
  /scratch/work/public/singularity/hudson/images/rocm4.5.2-ubuntu20.04.3.sif \
  /bin/bash -c "
 source /ext3/env.sh
 $args 
"