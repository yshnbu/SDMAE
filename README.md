## pytorch implementation for "SDMAE: A Self-supervised Learning Method Based on a Self-distillation Masked Autoencoder for Anomalous Sound Detection"

### Installation
---
```shell
$ conda create -n stgram_mfn python=3.7
$ conda activate sdmae
$ pip install -r requirements.txt
```

### Dataset
---
[DCASE2020 Task2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) Dataset: 
+ [development dataset](https://zenodo.org/record/3678171)
+ [additional training dataset](https://zenodo.org/record/3727685)
+ [Evaluation dataset](https://zenodo.org/record/3841772)

data path can be set in config.yaml


### Model Weights File
---
Our trained model weights file for loading can be get in [here](https://pan.baidu.com/s/1S2nuGVpvs_B6wq33OjuAtw?pwd=1111)

Then place it into ./runs/SDMAE/model.
```shell
$ python infer.py
```
### Model Weights File
---
Our work is inspired by the following projects:
+ [STgram-MFN](https://github.com/liuyoude/STgram-MFN)
+ [ConvMAE](https://github.com/Alpha-VL/ConvMAE)
+ [Dino](https://github.com/facebookresearch/dino)

