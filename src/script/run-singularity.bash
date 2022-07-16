#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi



singularity \
  exec --nv \
  --overlay /ocean/projects/cis220033p/ajn313/data/singularity/openclip_w0.ext3:ro \
  /ocean/projects/cis220033p/ajn313/data/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
 source /ext3/env.sh
 $args 
"
