# PIDNet with RPEM
## Introduction
This is the official repository for our recent work (Zhou, G. C., Cheng, C., & Chen, Y. Z. (2024). Efficient multi-branch segmentation network for situation awareness in autonomous navigation. Ocean Engineering, 302, 117741.).
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
* In config file, the option `space_block=True` represents the use of RPEM. The option `space_emb='linear'`, `space_emb='sinnn'` and `space_emb='sin'` represents the use of linear positional encoding, sin positional encoding and normalized sin positional encoding. 

### 2. Testing
* Follow the guide of [mmsegmentation](README_mmseg.md) or use basic usage as follows.
````bash
python tools/test.py configs/RPEM/pidnet-s-rpem-sin.py
````


