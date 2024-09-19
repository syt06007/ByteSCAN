# ICASSP2025_ByteSCAN
<br>
<p align="center"> <img src="https://github.com/syt06007/ByteSCAN/blob/main/images/ByteSCAN.png" width="100%"> </p>

***PyTorch implementation of our paper "Pushing the Limit in File Fragment Classification: Leveraging Deep Convolutional Features and Shallow Self-Attention on Fixed-Byte Segments". [ICASSP 2025 Under Review]***<br>

## News and Updates:
* 2024-09-20: Codes and models are uploaded.

## Preparation:
### 1. Requirement:
* PyTorch 2.2.1, torchvision 0.17.1. The code is tested with python=3.10, cuda=12.2

```
pip install -r requirements.txt
```

### 2. Datasets:
To use the [FFT75](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset) dataset for training and validation, follow these steps:

1. Download the **4k_1** data from the **FFT75** dataset.
2. Rename the downloaded files:
   - Rename `train.npz` to `train_4096.npz`
   - Rename `val.npz` to `val_4096.npz`
3. Place the renamed files in the following directory:
   - `dataset/fft/`
4. Run the data split script to organize the dataset into the desired structure. Adjust the arguments in `data_split.py` as needed to ensure the files are split and organized according to the following path structure:

### 3. Backbone Pretraining:
This section explains how to train the **ResNet18-1D Backbone**. The training process allows you to select one of three options for pretraining using the parser. You can choose between the following settings:

- `fft_RandomCrop`
- `fft_FixedCrop`
- `govdocs_RandomCrop`

To train the backbone with one of these options, run the following command:

```bash
python backbone_pretrain.py --settings=fft_RandomCrop


<br/>

If everything is prepared, you will have the following data structure

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

## Parser Options

This project includes a parser with predefined options to facilitate the reproduction of results presented in the paper. 

- The `--settings` option allows you to choose from various configurations, such as `set_num_1`, `set_num_2`, etc., each corresponding to a specific experiment setup.
- The `--block_size` option lets you specify the size of data segments, with available options being `512`, `1024`, `2048`, and `4096`.


## Train/Test:
```
python main.py --settings=set_num_1 --block_size=4096
```

```
python test.py --settings=set_num_1 --block_size=4096
```
* Checkpoint will be saved to `./ckpt/fft(or govdocs)/`.


## Reproduce the parameters reported in our paper:
* Run `models/transformer.py` to reproduce the parameters reported in our paper. Note that, the parameter measurement require `pip install thop`
```
python models/transformer.py
```

## Results:

### Performance on Datasets:
<p align="center"> <img src="https://github.com/syt06007/ByteSCAN/blob/main/images/results.png" width="100%"> </p>


<!-- ## Citiation
**If you find this work helpful, please consider citing:**
```
@InProceedings{,
    author    = {},
    title     = {},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
```
<br> -->


## Contact
**Welcome to email to [khs06007@hanyang.ac.kr](khs06007@hanyang.ac.kr) for any question regarding this work.**

## Reference
* The implementation of other models referenced in this paper can be found in **[[XMP_TIFS](https://github.com/DominicoRyu/XMP_TIFS)]**.

## License
A patent application for ByteSCAN has been submitted and is under review for registration. ByteSCAN is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.
