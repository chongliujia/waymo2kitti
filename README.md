# waymo2kitti


## Setup
### Step1. Create and activate a new virtual environment 
```
conda create -n waymo-kitti python=3.8
conda activate waymo-kitti
```
### Step2. Install TensorFlow
```
pip install tensorflow==2.6.0
```

### Step3. Install waymo open dataset precompiled packages
```
pip install waymo-open-dataset-tf-2-11-0==1.6.1
```
### Step4. Install other packages

```
pip install opencv-python matplotlib tqdm open3d
```
## Convert WOD-format data to KITTI-format data
### Step1. Download Waymo Datasets
[Waymo open datasets](https://waymo.com/open/download/)

Choose Waymo Perception Datasets 

v1.4.1, December 2022: Improved the quality of the 2D video panoptic segmentation labels

### Step2. Use waymo2kitti.py
For example

```
python waymo2kitti.py yourpath/waymo/training yourpath/waymo2kitti/training --prefix "training" --num_proc [your number_proc]
python waymo2kitti.py yourpath/waymo/validation yourpath/waymo2kitti/training --prefix "training" --num_proc [your number_proc]
python waymo2kitti.py yourpath/waymo/testing yourpath/waymo2kitti/testing --prefix "testing" --num_proc [your number_proc]

```

## References
1. [Waymo-KITTI Converter](https://github.com/caizhongang/waymo_kitti_converter)
2. [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)