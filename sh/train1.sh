OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6 python sh/baseline.py \
    --sf vgg16+NetVLAD_prepitts_b1_neg10_trainall_customnorm_3- --port 8814\
    --image_size 512 512 --batch 1 --workers 4 \
    --normalize_mean 0.485 0.456 0.406 --normalize_std 0.229 0.224 0.225 \
    --ckpt_path ./jobs/capstone2020/20200728-055558_vgg16+NetVLAD_prepitts_b1_neg10_trainall_customnorm_0-2/ckpt/checkpoint_subset000_epoch002.pth.tar \
    --train --tuple --neg_num 10 \
    --cluster 64 --topk 10 --dataset 2 --optimizer 0 --scheduler 1 \
    --extractor 2 --searching 1 --metric 0 --positive_selection 0