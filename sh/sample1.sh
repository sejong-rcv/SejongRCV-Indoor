OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python sh/baseline.py \
    --sf debug --port 8823 \
    --image_size 512 512 --batch 1 --workers 4 \
    --ckpt_path ./jobs/capstone2020/20200527-015656_D2+NetVLAD_top10/ckpt/checkpoint_subset000_epoch024.pth.tar \
    --valid_sample --cluster 64 --pose_estimation --pose_ld 4 --topk 10 --dataset 2 --optimizer 0 --scheduler 0 \
    --extractor 2 --searching 1 --metric 0 --positive_selection 0 \
    --qualitative