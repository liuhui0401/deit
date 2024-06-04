
srun -p Gvlab-S1-32 --gres=gpu:8 --cpus-per-task 8 --ntasks-per-node=8 --quotatype=auto --job-name T2I_gen \
torchrun --master_port=28391 --nproc_per_node 8 main.py \
--model DiT_Llama_80M_avg --cached_data \
--batch-size 256 --data-path ./data --output_dir results/debug \
2>&1 | tee -a results/debug/output.log

#--model DiT_Llama_600M_patch2 deit_small_patch16_224 \

# 0.9.2