#!/bin/bash
#SBATCH -J stride-K80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4   # to ensure we run on first gpu on system
#SBATCH --time=1:00:00
#SBATCH --mem=12000M # gpu2
#SBATCH --partition=gpu2
#SBATCH --exclusive
#SBATCH --array 1-4
#SBATCH -o slurmgpu2_array-%A_%a.out
#SBATCH -e slurmgpu2_array-%A_%a.err


k=$SLURM_ARRAY_TASK_ID

APP_ROOT=${HOME}/cuda-workspace/cuda-stride-benchmark
APP_BIN_DIR=${APP_ROOT}/release
CUDA_VER=9.0.176
#CLOCK=562
CLOCK=823

module purge
module load gcc/5.3.0 cuda/${CUDA_VER}

RESULT_DIR=$APP_ROOT/results/K80/cuda-${CUDA_VER}
mkdir -p ${RESULT_DIR}

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

FILENAME=${RESULT_DIR}/${APP_NAME}-${CLOCK}mhz.csv


# call app
srun --cpu-freq=medium --gpufreq=2505:${CLOCK} ${APP_BIN_DIR}/${APP_NAME} $((1*1048576)) $((1024*1048576)) > ${FILENAME}
# check if clocks have been throttled
nvidia-smi -i 0 -q -d PERFORMANCE
