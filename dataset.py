import torch
import os
import numpy as np

from einops import rearrange
from typing import Literal

class SCAN_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type:    Literal["fft", "govdocs"], 
                block_size:             Literal[4096, 2048, 1024, 512], 
                mode:                   Literal["train", "val"], 
                kernel_size:int=512, 
                overlap:bool=True
                ):



        self.kernel_size = kernel_size
        if overlap:
            self.stride = int(self.kernel_size/2)
        else: # Non-overlapping setup
            self.stride = int(self.kernel_size)

        dir = 'dataset'
        data_path = os.path.join(dir + f'/{dataset_type}/{mode}_{block_size}.npz')
        data_np = np.load(data_path)

        print('Data Path : ', data_path)

        data = data_np['x']
        label = data_np['y']

        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)


    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):

        x = self.x_data[idx].long()
        y = self.y_data[idx].long()

        if x.shape[0]==512:
            x_arr = rearrange(x, '(p n) -> n p', p=512, n=1)
        else:
            x_arr = x.unfold(0, self.kernel_size, self.stride)

        return x_arr, y


class SCAN_Dataset_VarSize(torch.utils.data.Dataset):
    def __init__(self, dataset_type:str='fft', 
                mode:str='val',
                kernel_size:int=512,
                overlap:bool=True
                ):

        self.kernel_size = kernel_size
        if overlap:
            self.stride = int(self.kernel_size/2)
        else: # Non-overlapping setup
            self.stride = int(self.kernel_size)

        data_512 = np.load(os.path.join(dir + f'/{dataset_type}/{mode}_512.npz')['x'])
        data_1024 = np.load(os.path.join(dir + f'/{dataset_type}/{mode}_1024.npz')['x'])
        data_2048 = np.load(os.path.join(dir + f'/{dataset_type}/{mode}_2048.npz')['x'])
        label = (os.path.join(dir + f'/{dataset_type}/{mode}_512.npz')['y'])
        
        self.x_512 = torch.from_numpy(data_512)
        self.x_1024 = torch.from_numpy(data_1024)
        self.x_2048 = torch.from_numpy(data_2048)
        self.y_data = torch.from_numpy(label)
        
    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        y = self.y_data[idx].long()
    
        x_crop_512 = self.x_512[idx].long()
        x_crop_1024 = self.x_1024[idx].long()
        x_crop_2048 = self.x_2048[idx].long()

        x512 = x_crop_512.unfold(0, self.kernel_size, self.stride)
        x1024 = x_crop_1024.unfold(0, self.kernel_size, self.stride)
        x2048 = x_crop_2048.unfold(0, self.kernel_size, self.stride)
        
        return x512, x1024, x2048, y


class BackboneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type:    Literal["fft", "govdocs"], 
                mode:                   Literal["train", "val"],
                is_fixed:               Literal[True, False]):
        self.is_fixed = is_fixed

        dir = 'dataset'
        data_path = os.path.join(dir + f'/{dataset_type}/{mode}_4096.npz')
        data_np = np.load(data_path)

        print('Data Path : ', data_path)

        data = data_np['x']
        label = data_np['y']

        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)


    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        if self.is_fixed:
            rand_idx = 0 # fix 0:512
        else:
            rand_idx = np.random.randint(0, 4096-512)

        x = self.x_data[idx].long()
        y = self.y_data[idx].long()

        x = x[rand_idx:rand_idx+512]

        return x, y
