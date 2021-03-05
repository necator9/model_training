# Training a classification model for the [lightweight detection algorithm](https://github.com/necator9/detection_method)  
The following steps describe the process of synthetic features and noises generation, and training a logistic regression classifier.

![Training pipeline](doc/img/overall_block_diagram.png)

## 1. [Camera calibration](https://github.com/necator9/video2calibration)
## 2. [Camera matrix optimization](https://github.com/necator9/get_optimal_cam_mtx)
## 3. Generate synthetic features using 3D models of objects 
Synthetic scenes containing 3D objects are rendered in the 2D image. 
Parameters of objects (size, rotation, coordinates) and camera (incline, intrinsics) are set in the config file.
These parameters are chosen according to expected usage scenario.

### 3.1 Install requirements
```text
pip3 install -r requirements.txt
```

### 3.2 Adjust config file

Parameters of a scene to be generated are specified in yaml file (see examples `scenes/all_scenes.yml` and `scenes/lamp_pole.yml`).
#### Section `camera`
Parameters in this section configure 
Important keys:

| key | description |
|---|---|
| params | path to optimized camera parameters which have been obtained on steps 1 and 2 (mentioned above ), see `camera_matrices/` for examples|
| angle  |  camera incline towards to ground surface: 0 deg. - the camera is parallel to the ground surface; -90 deg. - camera points perpendicularly down |
| x_range | object coordinates in meters along x axis (left, right relatively camera)  |
| y_range  | ground surface coordinates in meters relatively to a camera origin (e.g. -5 is 5m of camera height)  |
| z_range | distance to an object from camera in meters |
| thr | size of a kernel for morphological dilate on the resulting mask to imitate motion blur |
| rotate_y | range of object rotation angles about y axis (movement direction imitation) |
| ry_init| initial offset of r_y (some objects are initially rotated by back to a camera) | 
| scale | scaling the object along y axis (the desired height  of the object in meters), the scaling is uniform |

### 3.2 Start features generation
It is recommended to perform features generation on multi-core servers, otherwise the process can take significant time.

```text
usage: feat_gen.py [-h] [--csv CSV] [--show] config

Generate features of 3D objects

positional arguments:
  config      path to the configuration file

optional arguments:
  -h, --help  show this help message and exit
  --csv CSV   path to the output csv file (default:features.csv)
  --show      show the generated images (for debug only)
```




## 3. Generate noises
Pass the path to generated features as CL argument and run the script to generate noises.

### Usage 
```text
usage: noise_gen.py [-h] [-n NOISES] [-p POINTS] features

Generate noises around features

positional arguments:
  features              path to the features csv file

optional arguments:
  -h, --help            show this help message and exit
  -n NOISES, --noises NOISES
                        path to the output csv file containing noises features (default:noises.csv )
  -p POINTS, --points POINTS
                        amount of points per hull (default: 40000)
```

## 4. Train logistic regression classifier
Pass the path to generated features and noises as CL arguments and run the script.

### Usage 
```text
usage: train_separate_models.py [-h] [-c CLF] features noises

Train the logistic regression classifier

positional arguments:
  features           path to the features csv file
  noises             path to the noises csv file

optional arguments:
  -h, --help         show this help message and exit
  -c CLF, --clf CLF  path to the output classifier (default: clf.pcl)
```