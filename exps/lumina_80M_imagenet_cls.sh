#!/bin/bash
#SBATCH -p Gvlab-S1-32
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:4
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode append
#SBATCH --job-name=Lumina_cls_80M

exps=80M_cls_correct_sampler_lr2.5e-4_gpu4
mkdir -p results/${exps}

torchrun --nproc_per_node 4 main.py \
--model DiT_Llama_80M_patch2 \
--lr 2.5e-4 \
--resume results/${exps}/checkpoint.pth --cached_data \
--batch-size 256 --data-path ./data --output_dir results/${exps} \
2>&1 | tee -a results/${exps}/output.log


