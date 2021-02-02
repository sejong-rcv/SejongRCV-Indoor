# SejongRCV-Indoor

This repository contains the implementation of the solution of indoor 3rd rank team SejongRCV-Indoor in NAVER LABS Mapping & Localization Challenge 
<br/><br/><br/><br/>

## Publication

3rd Place Solution to NAVER LABS Mapping & Localization Challenge 2020: Indoor Track (IPIU2021)
- Published Paper : [[Link](https://github.com/sejong-rcv/SejongRCV-Indoor/files/5911122/default.pdf)]
- Modified Paper(Modified in Table 4) : [[Link](https://github.com/sejong-rcv/SejongRCV-Indoor/files/5911129/3rd.Place.Solution.to.NAVER.LABS.Mapping.and.Localization.Challenge.2020.Indoor.Track.pdf)] 
- Presentation materials : [[Link](https://github.com/sejong-rcv/SejongRCV-Indoor/files/5911387/IPIU.pdf)]
- Presentation video : [[Link](https://youtu.be/J5143Ct6Tdo)]

<br/><br/><br/><br/>

## Installation

1. **Get the code.** 
```
  git clone https://github.com/sejong-rcv/SejongRCV-Indoor.git
```

2. **We provide a Docker images.**

```
docker pull clockjw/naver-ml-challenge
```
<br/><br/><br/><br/>


## Dataset  
  
Running this code requires M/L Challenge 2020 Dataset. (available [here](https://challenge.naverlabs.com/))  
The data path can be checked below.

``` 
<DATA_PATH>

+-- NaverML_indoor
|   +-- [floor]
|   |   +-- train
|   |   |   +-- [date]
|   |   |   |   +-- camera_paramters.txt
|   |   |   |   +-- groundtruth.hdf5
|   |   |   |   +-- map.pcd
|   |   |   |   +-- images
|   |   |   |   |   |   +-- [camid]_[timestamp].jpg
|   |   |   |   +-- pointclouds_data
|   |   |   |   |   |   +-- [lidarid]_[timestamp].pcd
|   |   |   +-- PointCloud_all
|   |   |   |   +-- map.pcd
|   |   |   +-- csv
|   |   |   |   +-- v2
|   |   |   |   |   |   +-- train_all
|   |   |   |   |   |   |    +-- train_[floor].csv 
|   |   |   |   |   |   +-- train_val
|   |   |   |   |   |   |    +-- train_[floor].csv 
|   |   |   |   |   |   |    +-- val_[floor].csv 
|   |   |   |   |   |   |    +-- val_[floor]_sample10.csv 
|   |   +-- test
|   |   |   +-- date
|   |   |   |   +-- camera_paramters.txt
|   |   |   |   +-- groundtruth.hdf5
|   |   |   |   +-- images
|   |   |   |   |   |   +-- [camid]_[timestamp].jpg
|   |   |   +-- csv
|   |   |   |   +-- test_[floor].csv
```
<br/><br/><br/><br/>

## Image retrieval
- **Train only**
```
python baseline.py --image_size 512 512 --batch 1 --shuffle --workers 8 \ 
                   --save_folder [SAVE_FOLDER] --train --cluster 64 --tuple \
                   --positive_selection 0 --dataset [DATASET_FLAG] --optimizer 0 --scheduler 1 \
                   --extractor [EXTRACTOR_FLAG] --searching [SEARCHING_FLAG] --metric 0
```

- **Train & Vaild**
```
python baseline.py --image_size 512 512 --batch 1 --shuffle --workers 8 \
                   --save_folder [SAVE_FOLDER] --train --cluster 64 --tuple \
                   --positive_selection 0 --dataset [DATASET_FLAG] --optimizer 0 --scheduler 1 \
                   --extractor [EXTRACTOR_FLAG] --searching [SEARCHING_FLAG] --metric 0 --valid
```
 
And if you want to use smaller validation sampling set, use "--valid_sample" instead of "--valid"


- **Test**
```
python baseline.py --image_size 512 512 --batch 1 --shuffle --workers 8 \
                   --save_folder [SAVE_FOLDER] --test --dataset [DATASET_FLAG] --extractor [EXTRACTOR_FLAG] \
                   --searching [SEARCHING_FLAG] --metric 0
```

- **[EXTRACTOR_FLAG]**
  - 0 : D2_NetVLAD (D2 means VGG16 provided by [D2Net](https://github.com/mihaidusmanu/d2-net))
  - 1 : Pitts_NetVLAD (Piits means VGG16 provided by [Nanne/pytorch-NetVLAD](https://github.com/Nanne/pytorch-NetVlad)
  - 2 : APGeM
  - 3 : APGeM_LM18
  - 4 : Ensemble(APGeM + APGeM_LM18 + D2_NetVLAD)
  - 5 : Ensemble(APGeM + APGeM_LM18 + Pitts_NetVLAD)  
  - 6 : Ensemble(APGeM + APGeM_LM18)  
  - 7 : Ensemble(APGeM + D2_NetVLAD)  
  - 8 : Ensemble(APGeM + Pitts_NetVLAD)  
  - 9 : Ensemble(APGeM_LM18 + D2_NetVLAD)  
  - 10 : Ensemble(APGeM_LM18 + Pitts_NetVLAD)  

If you want to train handcraft extractor, use below instead of "--extractor"

```
---handcraft [HANDCRAFT_FLAG]
```

- **[HANDCRAFT_FLAG]**
  - 0 : SIFT+VLAD
  - 1 : rootSIFT+VLAD


- **Trained weight**
  - [checkpoint](https://drive.google.com/file/d/1M_FRhv7Md7qYFh8p3CpyELwUjijN9HyR/view?usp=sharing)
  - Place it like this
  ```
  +-- arxiv
  |   +-- tar or pth or pt or pkl (contents of zip)
  ```


<br/><br/><br/><br/>

## Reranking

```
python baseline.py --image_size 512 512 --batch 1 --shuffle --workers 8 \
                   --save_folder [SAVE_FOLDER] --train --cluster 64 --tuple \
                   --positive_selection 0 --dataset [DATASET_FLAG] --optimizer 0 --scheduler 1 \
                   --extractor [EXTRACTOR_FLAG] --searching [SEARCHING_FLAG] --metric 0 --valid \
                   --pose_ld [POSE_LD_FLAG] --rerank [RERANK_FLAG]
```

if you want to use lmr, please add argument "--lmr_score"
<br/><br/><br/><br/>


## Pose estimation

```
python baseline.py --image_size 512 512 --batch 1 --shuffle --workers 8 \
                   --save_folder [SAVE_FOLDER] --train --cluster 64 --tuple \
                   --positive_selection 0 --dataset [DATASET_FLAG] --optimizer 0 --scheduler 1 \
                   --extractor [EXTRACTOR_FLAG] --searching [SEARCHING_FLAG] --metric 0 --valid \
                   --pose_ld [POSE_LD_FLAG] --pose_estimation --pose_noniter
```
if you want to use saved pointcloud, please add argument "--pose_pointcloud_load"  
if you want to stack (n) frames, please add argument "--pose_covisibility"  
if you want to use cupy, please add argument "--pose_cupy"
<br/><br/><br/><br/>


## Reference
[1] https://challenge.naverlabs.com/  
[2] https://github.com/Relja/netvlad  
[3] https://github.com/Nanne/pytorch-NetVlad  
[4] https://github.com/lyakaap/NetVLAD-pytorch  
[5] https://github.com/magicleap/SuperGluePretrainedNetwork  
[6] https://github.com/mihaidusmanu/d2-net  
[7] https://github.com/almazan/deep-image-retrieval  
 
