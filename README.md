# SejongRCV-Indoor

This repository contains the implementation of the solution of indoor 3rd rank team SejongRCV-Indoor in NAVER Map & Localization Challenge 

### Installation

1. Get the code. 
```
  git clone https://github.com/sejong-rcv/SejongRCV-Indoor.git
```

2. We provide a Docker images.

```
docker pull clockjw/naver-ml-challenge
```

### Dataset

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

[checkpoint](https://drive.google.com/file/d/1q7uvGpmsJevyG99uvG_8on91jUDKBWvr/view?usp=sharing)
