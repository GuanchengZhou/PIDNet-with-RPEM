# PIDNet with RPEM
## Introduction
This is the official repository for our recent work (in review).
## Usage
### 0. Prepare the dataset
* Follow the guide of [mmsegmentation](README_mmseg.md) to prepare the environment.
* Download the [MaSTr1325](https://box.vicos.si/borja/viamaro/index.html) or contact us for experiental data in data/.
* Transform the format of the dataset to Cityscapes format.
### 1. Training
* Use the config file ([configs/RPEM/pidnet-s-rpem-sin.py](configs/RPEM/pidnet-s-rpem-sin.py)) and follow the guide of [mmsegmentation](README_mmseg.md) to train the model. The basic usage is as follows.
````bash
python tools/train.py configs/RPEM/pidnet-s-rpem-sin.py
````
### 2. Testing
Follow the guide of [mmsegmentation](README_mmseg.md) or use basic usage as follows.
````bash
python tools/test.py configs/RPEM/pidnet-s-rpem-sin.py
````


