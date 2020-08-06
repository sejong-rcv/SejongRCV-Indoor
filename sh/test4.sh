OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 python sh/baseline.py \
    --sf debug --port 8823 \
    --image_size 512 512 --batch 1 --workers 4 \
    --test --cluster 64 --pose_estimation --pose_ld 4 --topk 10 --dataset 2 --optimizer 0 --scheduler 0 \
    --extractor 2 --searching 1 --metric 0 --positive_selection 0 \
    --pose_pointcloud_load \
    --pose_covisibility 1 --pose_noniter --lmr_score 0.5 --rerank 0


    #    --normalize_mean 0.485 0.456 0.406 --normalize_std 0.229 0.224 0.225 \


