# SRGAN-from-scratch
This project is an implementation of SRGAN model in the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) using Pytorch. 
(Note: The perceptual loss in this project also include Total Variation Loss beside Content Loss & Adversarial Loss in the original paper)

### Setting Up: Hyperparams, Dataset paths, ... can be modified in [utils/config.py]

### Training:
##### 1, Dataset: 
- Training: [DF2k](https://www.kaggle.com/datasets/anvu1204/df2kdata) (used for pretrain this project)
- Validation: [SR Benchmarks](https://www.kaggle.com/datasets/jesucristo/super-resolution-benchmarks)
##### 2, Training:
```cmd
python model/train.py
```

### Testing:
##### 1, Pretrained Model:
- Generator: [Link](https://drive.google.com/file/d/1xLbD_NzM-QC0exkdGCznUy5-CZ4BJUbs/view?usp=drive_link)
- Discriminator: [Link](https://drive.google.com/file/d/1vIPPjmsyHlebEiw_28liA0XpU_-jJfwA/view?usp=drive_link)
##### 2, Inference Single Image:
```cmd
python infer.py --img_path
```
(Note: The default output directory is [infer_results])
