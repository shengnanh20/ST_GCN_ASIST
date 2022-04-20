# Predicting Team Performance with Spatial Temporal Graph Convolutional Networks

This repo contains the code related to the paper "Predicting Team Performance with Spatial Temporal Graph Convolutional Networks" we submitted to ICPR 2022.

## Getting Started

### Dependencies

* Python 3.7
* Pytorch 1.9
* torch-geometric-temporal

### Dataset
To get dataset with different node feature or length, you could run 'file_processing.py'.

### Usage

To train and test the proposed model:
```
python main.py
```

To train and test the backbone models:
```
python backbones.py
```
### License
This code is released under the UCF license.
