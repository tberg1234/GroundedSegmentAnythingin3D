#!/usr/bin/env bash

module load cuda

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python run_seg_gui.py --config="configs/llff/seg/seg_fern.py" --segment --sp_name="_gui" --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type "seg_img" "seg_density"