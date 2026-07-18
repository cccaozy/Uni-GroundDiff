CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch  --master_port 14321 --nproc_per_node 6 --nnodes 1 generate_png.py
