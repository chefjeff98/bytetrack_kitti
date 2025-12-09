## ByteTrack evaluated on KITTI dataset

This repo contains the code used to evaluate the KITTI MOT training dataset on ByteTrack. 

PLEASE NOTE: This repo was created for class NAVARCH 565 at the University of Michigan. If you are not a student or faculty member, you will not have access to the drive files. 

### Installation
Step1. Install modified ByteTrack on host machine
```shell
git clone https://github.com/chefjeff98/bytetrack_kitti.git
cd ByteTrack
pip install -r requirements.txt
pip install -e .
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip install cython_bbox
```

### Download large files
Download the files at https://drive.google.com/drive/folders/1Trchno13FVShx5xxQ0k1kHGwW3zXOXL6?usp=sharing and add them to your ByteTrack directory. Access is granted only to University of Michigan students and faculty. 

### Run and evaluate ByteTrack
These commands will run object detection, tracking, and evaluation for ByteTrack, SORT, DeepSORT, and MOTDT.
```shell
python tools/track.py -f .\exps\yolox_s_kitti.py -c .\pretrained\yolox_s.pth --fp16 --fuse --test -b 1 --track_thresh 0.6
python tools/track_sort.py -f .\exps\yolox_s_kitti.py -c .\pretrained\yolox_s.pth --fp16 --fuse --test -b 1 --track_thresh 0.6
python tools/track_deepsort.py -f .\exps\yolox_s_kitti.py -c .\pretrained\yolox_s.pth --fp16 --fuse --test -b 1 --track_thresh 0.6
python tools/track_motdt.py -f .\exps\yolox_s_kitti.py -c .\pretrained\yolox_s.pth --fp16 --fuse --test -b 1 --track_thresh 0.6
```
