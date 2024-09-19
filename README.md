# ICASSP2025_ByteSCAN
<br>
<!-- <p align="center"> <img src="" width="90%"> </p> -->

***PyTorch implementation of our paper "Pushing the Limit in File Fragment Classification: Leveraging Deep Convolutional Features and Shallow Self-Attention on Fixed-Byte Segments". [ICASS 2025 Under Review]***<br>

## News and Updates:
* 2024-09-20: Codes and models are uploaded.

## Preparation:
### Requirement:
* PyTorch 2.3.0, torchvision 0.4.1. The code is tested with python=3.10, cuda=12.2

```
pip install -r requirements.txt
```

### Datasets:

### Path structure:
  ```
    ByteSCAN/
    ├── ckpt/
    │   ├── backbone/
    │   │   ├── FFT75_FixedCrop_ResNet18.pth
    │   │   ├── FFT75_RandomCrop_ResNet18.pth
    │   │   ├── Govdocs_RandomCrop_ResNet18.pth
    │   ├── fft/
    │   │   ├── .gitkeep
    │   ├── govdocs/
    │   │   ├── .gitkeep
    ├── dataset/
    │   ├── fft/
    │   │   ├── train_512.npz
    │   │   ├── train_1024.npz
    │   │   ├── train_2048.npz
    │   │   ├── train_4096.npz
    │   │   ├── val_512.npz
    │   │   ├── val_1024.npz
    │   │   ├── val_2048.npz
    │   │   ├── val_4096.npz
    │   ├── govdocs/
    │   │   ├── train_512.npz
    │   │   ├── train_1024.npz
    │   │   ├── train_2048.npz
    │   │   ├── train_4096.npz
    │   │   ├── val_512.npz
    │   │   ├── val_1024.npz
    │   │   ├── val_2048.npz
    │   │   ├── val_4096.npz
  ```

## Hyperparameter Settings

This project provides predefined settings that can be used to reproduce the results presented in the paper. Each setting is labeled as `set_num_1`, `set_num_2`, etc., and corresponds to specific configurations used during different experiments.

### `set_num_1 (Train/Test)`
This setting is the default configuration. It uses the **FFT75 dataset**, employs a **pretrained backbone (FFT75) **, and trains the **attention layer**.

### `set_num_2 (Train/Test)`
This setting is the default configuration. It uses the **GovDocs dataset**, employs a **pretrained backbone (govdocs) **, and trains the **attention layer**.

### `set_num_3 (Only Test)`
This setting is used for **Ablation experiments**. It uses a **pretrained model on 4096-byte data** and performs predictions on datasets of **various sizes (512, 1024, 2048)**

### `set_num_4 (Only Test)`
This setting is used for **Ablation experiments**. It uses a **pretrained backbone** and performs predictions using soft voting.

### `set_num_5 (Train/Test)`
This setting is used for **Ablation experiments**. It trains the attention layer using a backbone **pretrained with data fixed-cropped from a single position** instead of the backbone pretrained with randomly cropped data from 4096-byte data.

### `set_num_6 (Train/Test)`
This setting is used for **Ablation experiments**. It does not perform overlapping between segments (512 byte).

### `set_num_7 (Train/Test)`
This setting is used for **Ablation experiments**. It does not perform overlapping between segments (256 byte).


## Train/Test:
```
python main.py --settings=set_num_1
```

```
python test.py --settings=set_num_1
```
* Checkpoint will be saved to `./ckpt/fft(or govdocs)/`.

## Reproduce the scores on the HCI 4D LF benchmark:


## Reproduce the inference time reported in our paper:
* Run `models/transformer.py` to reproduce the parameters reported in our paper. Note that, the parameter measurement require `pip install thop`
```
python models/transformer.py
```

## Results:

### Performance on Datasets:
<!-- <p align="center"> <img src="https://raw.github.com/YingqianWang/OACC-Net/master/Figs/QuantitativeMSE.png" width="95%"> </p> -->


## Citiation
**If you find this work helpful, please consider citing:**
<!-- ```
@InProceedings{,
    author    = {},
    title     = {},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
``` -->
<br>

## Contact
**Welcome to email to [khs06007@hanyang.ac.kr](khs06007@hanyang.ac.kr) for any question regarding this work.**
