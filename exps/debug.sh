


exps=debug
mkdir -p results/${exps}

torchrun --nproc_per_node 1 main.py \
--model DiT_Llama_80M_noise \
--resume results/${exps}/checkpoint.pth \
--batch-size 256 --data-path /data0/data/imagenet/ --output_dir results/${exps} \
2>&1 | tee -a results/${exps}/output.log
