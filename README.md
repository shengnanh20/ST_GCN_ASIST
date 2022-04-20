# Predicting Team Performance with Spatial Temporal Graph Convolutional Networks

This repo contains the code related to the paper "Predicting Team Performance with Spatial Temporal Graph Convolutional Networks" we submitted to ICPR 2022.


### Dependencies

* Python 3.7
* Pytorch 1.9
* torch-geometric-temporal

### Dataset
To generate player features from the metadata and FoV files, you could run 'file_processing.py'. Also, you could modify the parameters in it to change the length of the input sequence (15/30/60 seconds).

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
