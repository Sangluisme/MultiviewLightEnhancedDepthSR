# MultiviewLightEnhancedDepthSR

This code implements the approach for the following [research paper](https://vision.in.tum.de/_media/spezial/bib/sang2020wacv.pdf):
>**Inferring Super-Resolution Depth from a Moving Light-Source Enhanced RGB-D Sensor: A Variational Approach**
>*Sang, L., Haefner, B. and Cremers, D.;Winter Conference on Applications of Computer Vision (WACV) 2020*
> *Spotlight Presentation*  
![alt tag](https://vision.in.tum.de/_media/spezial/bib/sang2020wacv.png)

We present a novel approach towards depth map super-resolution using multi-view uncalibrated photometric stereo is presented. Practically, an LED light source is attached to a commodity RGB-D sensor and is used to capture objects from multiple viewpoints with unknown motion. This nonstatic camera-to-object setup is described with a nonconvex variational approach such that no calibration on lighting or camera motion is required due to the formulation of an end-to-end joint optimization problem. Solving the proposed variational model results in high resolution depth, reflectance and camera pose estimates, as we show on challenging synthetic and real-world datasets.

## 1. Requirements

This code has four third party dependencies:

0) MATLAB (Code was tested and works under MATLAB R2018b)
1) [inpaint_nans](https://de.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/4551/versions/2/download/zip) (mandatory)
2) [CMake](https://cmake.org/) (mandatory minimum required 2.8)
3) [OpenCV](https://opencv.org/) (mandatory)
4) [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) (mandatory)

## 2. Getting Started
### 0) compile DVO code
go to DVO folder, create a build folder then run 
```
camke ..
```
then 
```
make
```
This step will compile a mex function allow matlab to run c++ code, therefore make sure the matlab root is in the system path.

### 1) run matlab code
run main_mvps.m

this code will add data folder, src folder and DVO/build/ folder to matlab path, so if you name the folder differently, please change accordingly.



