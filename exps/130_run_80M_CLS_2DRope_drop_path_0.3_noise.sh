


exps=80M_CLS_2DRope_No_Cached_Dataset_Stransform_2.5e-4_gpu4_drop_path_0.3_noise
mkdir -p results/${exps}


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=28391 --nproc_per_node 4 main.py \
--model DiT_Llama_80M_noise \
--use_noise \
--lr 2.5e-4 \
--drop-path 0.3 \
--resume results/${exps}/checkpoint.pth \
--batch-size 256 --data-path /data0/data/imagenet/ --output_dir results/${exps} \
2>&1 | tee -a results/${exps}/output.log
