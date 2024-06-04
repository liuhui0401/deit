


exps=80M_CLS_2DRope_No_Cached_Dataset_Stransform_2.5e-4_gpu4
mkdir -p results/${exps}


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=28392 --nproc_per_node 4 main.py \
--model DiT_Llama_80M_avg \
--lr 2.5e-4 \
--resume results/${exps}/checkpoint.pth \
--batch-size 256 --data-path /data0/data/imagenet/ --output_dir results/${exps} \
2>&1 | tee -a results/${exps}/output.log
