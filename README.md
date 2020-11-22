# Generate synthetic features and train the logistic regression classifier

## 1. Generate synthetic features using 3D models of objects 
### Usage
```text
usage: feat_gen.py [-h] [--csv CSV] [--show] config

Generate features of 3D objects

positional arguments:
  config      path to the configuration file

optional arguments:
  -h, --help  show this help message and exit
  --csv CSV   path to the output csv file (default:features.csv )
  --show      show the generated images
```
### config.yml
Parameters of a scene to be generated are specified in yaml file (see `scenes/` for examples).

Important keys:

|   |   |
|---|---|
| angle  |  camera incline towards to ground surface: 0 deg. - the camera is parallel to the ground surface; -90 deg. - camera points perpendicularly down |
| x_range | object coordinates in meters along x axis (left, right relatively camera)  |
| y_range  | ground surface coordinates in meters relatively to a camera origin (e.g. -5 is 5m of camera height)  |
| z_range | distance to an object from camera in meters |
| thr | size of a kernel for morphological dilate on the resulting mask to imitate motion blur |
| rotate_y | range of object rotation angles about y axis (movement direction imitation) |
| ry_init| initial offset of r_y (some objects are initially rotated by back to a camera) | 
