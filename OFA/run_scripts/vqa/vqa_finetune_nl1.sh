#!/bin/bash
#SBATCH -J ofa
#SBATCH -A virtual_presenter
#SBATCH -p p100_normal_q
#SBATCH --nodes=1
#SBATCH -t 4:30:00
#SBATCH --gres=gpu:2
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 3 > $SLURM_JOBID.gpu.log &


# tried w v100_normal_q, but could not get resources
./train_vqa_base_distributed1.sh