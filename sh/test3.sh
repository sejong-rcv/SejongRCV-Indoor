OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 python sh/baseline.py \
    --sf test_b1_ens-6_notrain_trainall_pnpr10_undist_cov1_noniter --port 8823 \
    --image_size 512 512 --batch 1 --workers 4 \
    --test --cluster 64 --pose_estimation --pose_ld 4 --topk 10 --dataset 3 --optimizer 0 --scheduler 0 \
    --extractor 8 --searching 1 --metric 0 --positive_selection 0 \
    --pose_pointcloud_load --rerank 3 \
    --pose_covisibility 1 --pose_noniter \
    --topk_save top10_ens-6_pnpr10_b1_all.npy

    #    --normalize_mean 0.485 0.456 0.406 --normalize_std 0.229 0.224 0.225 \


