# Real-time Hand Gesture Recognition with 3D CNNs
PyTorch implementation of the article [Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks](https://arxiv.org/abs/1901.10323) and [Resource Efficient 3D Convolutional Neural Networks](https://arxiv.org/pdf/1904.02422.pdf), codes and pretrained models.


<div align="center" style="width:image width px;">
  <img  src="https://media.giphy.com/media/9M3aPvPOVxSQmYGv8p/giphy.gif" width=500 alt="simulation results">
</div>

Figure: A real-time simulation of the architecture with input video from EgoGesture dataset (on left side) and real-time (online) classification scores of each gesture (on right side) are shown, where each class is annotated with different color. 

## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pytorch torchvision cuda80 -c soumith
```

* Python 3

### Pretrained models
[Pretrained_models_v1 (1.08GB)](https://drive.google.com/file/d/11MJWXmFnx9shbVtsaP1V8ak_kADg0r7D/view?usp=sharing): The best performing models in [paper](https://arxiv.org/abs/1901.10323)

[Pretrained_RGB_models_for_det_and_clf (371MB)(Google Drive)](https://drive.google.com/file/d/1V23zvjAKZr7FUOBLpgPZkpHGv8_D-cOs/view?usp=sharing)
[Pretrained_RGB_models_for_det_and_clf (371MB)(Baidu Netdisk)](https://pan.baidu.com/s/114WKw0lxLfWMZA6SYSSJlw) -code:p1va

[Pretrained_models_v2 (15.2GB)](https://drive.google.com/file/d/1rSWnzlOwGXjO_6C7U8eE6V43MlcnN6J_/view?usp=sharing): All models in [paper](https://ieeexplore.ieee.org/document/8982092) with efficient 3D-CNN Models
## Preparation

### EgoGesture

* Download videos by following [the official site](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html).
* We will use extracted images that is also provided by the owners

* Generate n_frames files using ```utils/ego_prepare.py``` 

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

```bash
mkdir annotation_EgoGesture
python utils/ego_prepare.py training trainlistall.txt all
python utils/ego_prepare.py training trainlistall_but_None.txt all_but_None
python utils/ego_prepare.py training trainlistbinary.txt binary
python utils/ego_prepare.py validation vallistall.txt all
python utils/ego_prepare.py validation vallistall_but_None.txt all_but_None
python utils/ego_prepare.py validation vallistbinary.txt binary
python utils/ego_prepare.py testing testlistall.txt all
python utils/ego_prepare.py testing testlistall_but_None.txt all_but_None
python utils/ego_prepare.py testing testlistbinary.txt binary
```

* Generate annotation file in json format similar to ActivityNet using ```utils/egogesture_json.py```

```bash
python utils/egogesture_json.py 'annotation_EgoGesture' all
python utils/egogesture_json.py 'annotation_EgoGesture' all_but_None
python utils/egogesture_json.py 'annotation_EgoGesture' binary
```

### nvGesture

* Download videos by following [the official site](https://research.nvidia.com/publication/online-detection-and-classification-dynamic-hand-gestures-recurrent-3d-convolutional).

* Generate n_frames files using ```utils/nv_prepare.py``` 

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

```bash
mkdir annotation_nvGesture
python utils/nv_prepare.py training trainlistall.txt all
python utils/nv_prepare.py training trainlistall_but_None.txt all_but_None
python utils/nv_prepare.py training trainlistbinary.txt binary
python utils/nv_prepare.py validation vallistall.txt all
python utils/nv_prepare.py validation vallistall_but_None.txt all_but_None
python utils/nv_prepare.py validation vallistbinary.txt binary
```

* Generate annotation file in json format similar to ActivityNet using ```utils/nv_json.py```

```bash
python utils/nv_json.py 'annotation_nvGesture' all
python utils/nv_json.py 'annotation_nvGesture' all_but_None
python utils/nv_json.py 'annotation_nvGesture' binary
```

### Jester

* Download videos by following [the official site](https://20bn.com/datasets/jester).

* N frames and class index  file is already provided annotation_Jester/{'classInd.txt', 'trainlist01.txt', 'vallist01.txt'}

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

* Generate annotation file in json format similar to ActivityNet using ```utils/jester_json.py```

```bash
python utils/jester_json.py 'annotation_Jester'
```


## Running the code
* Offline testing (offline_test.py) and training (main.py)
```bash
bash run_offline.sh
```

* Online testing
```bash
bash run_online.sh
```

## Citations

Code, paper and repository originally created by Köpüklü, Gunduz, Kose and Rigoll.

```bibtex
@article{kopuklu_real-time_2019,
	title = {Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks},
	url = {http://arxiv.org/abs/1901.10323},
	author = {Köpüklü, Okan and Gunduz, Ahmet and Kose, Neslihan and Rigoll, Gerhard},
  year={2019}
}
```

```bibtex
@article{kopuklu2020online,
  title={Online Dynamic Hand Gesture Recognition Including Efficiency Analysis},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Gunduz, Ahmet and Kose, Neslihan and Rigoll, Gerhard},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  volume={2},
  number={2},
  pages={85--97},
  year={2020},
  publisher={IEEE}
}
```
