OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf rootSIFT+VLAD_top1 --train --valid --topk 1 --dataset 2 --handcraft 1 --searching 0 --metric 1

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf rootSIFT+VLAD_top10 --train --valid --topk 10 --dataset 2 --handcraft 1 --searching 0 --metric 1

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf rootSIFT+VLAD_top100 --train --valid --topk 100 --dataset 2 --handcraft 1 --searching 0 --metric 1

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf rootSIFT+VLAD_top1_locDeg --train --valid --topk 1 --dataset 2 --handcraft 1 --searching 0 --metric 0

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf rootSIFT+VLAD_top10_locDeg --train --valid --topk 10 --dataset 2 --handcraft 1 --searching 0 --metric 0

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf rootSIFT+VLAD_top100_locDeg --train --valid --topk 100 --dataset 2 --handcraft 1 --searching 0 --metric 0

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf SIFT+VLAD_top10_posefilter1_locDeg --train --valid --topk 10 --dataset 2 --handcraft 0 --searching 0 --metric 0 --pose_filter --posek 1 --pose_ld 1 --pose_matcher 1

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python baseline.py --port 8836 --image_size 256 256 --batch 1 --sf SIFT+VLAD_top10_posefilter1_ldRoot_locDeg --train --valid --topk 10 --dataset 2 --handcraft 0 --searching 0 --metric 0 --pose_filter --posek 1 --pose_ld 0 --pose_matcher 1


