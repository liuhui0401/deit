#!/bin/bash
#SBATCH -p Gvlab-S1-32
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:4
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode append
#SBATCH --job-name=Lumina_cls_80M_2DRope

exps=80M_cls_2DRope_lr2.5e-4_gpu4
mkdir -p results/${exps}

torchrun --master_port=28391 --nproc_per_node 4 main.py \
--model DiT_Llama_80M_avg --cached_data \
--lr 2.5e-4 \
--resume results/${exps}/checkpoint.pth \
--batch-size 256 --data-path ./data --output_dir results/${exps} \
2>&1 | tee -a results/${exps}/output.log


