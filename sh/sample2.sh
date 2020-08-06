OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python sh/baseline.py \
    --sf sample10s_pnv_qualitative --port 8823 \
    --image_size 512 512 --batch 1 --workers 8 \
    --valid_sample --cluster 64 --pose_estimation --pose_ld 4 --topk 10 --dataset 2 --optimizer 0 --scheduler 0 \
    --extractor 2 --searching 1 --metric 0 --positive_selection 0 \
    --pose_pointcloud_load --qualitative