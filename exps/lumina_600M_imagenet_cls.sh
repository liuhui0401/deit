#!/bin/bash
#SBATCH -p Gvlab-S1-32
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:8
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode append
#SBATCH --job-name=Lumina_cls_600M

exps=600M_cls
mkdir -p results/${exps}

torchrun --nproc_per_node 8 main.py \
--model DiT_Llama_600M_patch2 \
--batch-size 256 --data-path ./data --output_dir results/${exps} \
2>&1 | tee -a results/${exps}/output.log


