#!/bin/bash
#SBATCH -J stride-taurust1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=63000M # gpu2
#SBATCH --partition=test
#SBATCH --exclusive
#SBATCH --array 0-4
#SBATCH -o slurmgpu2_array-%A_%a.out
#SBATCH -e slurmgpu2_array-%A_%a.err


k=$SLURM_ARRAY_TASK_ID

APP_ROOT=${HOME}/cuda-workspace/cuda-stride-benchmark
APP_BIN_DIR=${APP_ROOT}/release
#CUDA_VER=9.0.176
CUDA_VER=9.2.88

module purge
module load gcc/5.3.0 cuda/${CUDA_VER}

export CUDA_DEVICE_ORDER=PCI_BUS_ID

for GPU in 0 1 2; do

    GPU_NAME=`nvidia-smi -i $GPU -q |grep 'Product Name'|awk '{print $5}'`
    RESULT_DIR=$APP_ROOT/results/${GPU_NAME}/cuda-${CUDA_VER}
    mkdir -p ${RESULT_DIR}

    if [ $k -eq 0 ]; then
        APP_NAME=reduction-cub
    fi
    if [ $k -eq 1 ]; then
        APP_NAME=saxpy-mono
    fi
    if [ $k -eq 2 ]; then
        APP_NAME=saxpy-grid
    fi
    if [ $k -eq 3 ]; then
        APP_NAME=reduction-mono
    fi
    if [ $k -eq 4 ]; then
        APP_NAME=reduction-grid
    fi

    if [ $GPU -eq 0 ]; then
#        CLOCK=1327
        CLOCK=1245
        nvidia-smi -i $GPU -ac 850,$CLOCK   # GV100
    elif [ $GPU -eq 1 ]; then
        CLOCK=1189
        nvidia-smi -i $GPU -ac 715,$CLOCK   # P100
    else
        CLOCK=1245
        nvidia-smi -i $GPU -ac 877,$CLOCK   # V100
    fi
    FILENAME=${RESULT_DIR}/${APP_NAME}-${CLOCK}mhz.csv

    # call app
    export CUDA_VISIBLE_DEVICES=$GPU
    ${APP_BIN_DIR}/${APP_NAME} $((1*1048576)) $((1024*1048576)) > ${FILENAME}

    # check if clocks have been throttled
    nvidia-smi -i $GPU -q -d PERFORMANCE
    nvidia-smi -rac -i $GPU
done
