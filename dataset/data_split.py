import numpy as np
seed = 0
np.random.seed(seed)


dataset_type = 'fft' 
mode = 'train' # train or val


data = np.load(f'dataset/{dataset_type}/{mode}_4096.npz')

x = data['x']
y = data['y']

list_512 = []
list_1024 = []
list_2048 = []

for idx, i in enumerate(range(len(y))):
    random_idx1 = np.random.randint(4096 - 512)
    random_idx2 = np.random.randint(4096 - 1024)
    random_idx3 = np.random.randint(4096 - 2048)
    
    x_512 = x[i][random_idx1:random_idx1+512]
    x_1024 = x[i][random_idx2:random_idx2+1024]
    x_2048 = x[i][random_idx3:random_idx3+2048]

    list_512.append(x_512)
    list_1024.append(x_1024)
    list_2048.append(x_2048)

data_512 = np.stack(list_512)
data_1024 = np.stack(list_1024)
data_2048 = np.stack(list_2048)

np.savez(f'dataset/{dataset_type}/{mode}_512.npz', x=data_512, y=y)
np.savez(f'dataset/{dataset_type}/{mode}_1024.npz', x=data_1024, y=y)
np.savez(f'dataset/{dataset_type}/{mode}_2048.npz', x=data_2048, y=y)