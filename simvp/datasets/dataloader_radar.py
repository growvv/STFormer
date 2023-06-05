import torch
import numpy as np
from torch.utils.data import Dataset

import os
import numpy as np
import torch.utils.data as da

def pixel_to_dBZ_2(img):
    return img * 70.0


def dBZ_to_pixel_2(dBZ_img):
    return np.clip(dBZ_img / 70.0, a_min=0.0, a_max=1.0)

class RadarDataset(Dataset):
    """radar dataset"""
    def __init__(self, data_file, sample_shape=(20, 1, 256, 256), input_len=10):

        self.mean = 0
        self.std = 1

        f = np.load(data_file)
        d = f['arr_0']

        # transform dBZ to pixel, [0, 1]
        d = dBZ_to_pixel_2(d)
        # Reshape and select requested number of samples
        d = d.reshape((-1,) + sample_shape)
        # d = np.transpose(d, (0, 1, 3, 4, 2))

        d = d.astype(np.float32)

        self.data = d
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index]
        return sample[:10], sample[10:]


def load_data(batch_size, val_batch_size, data_root,
              num_workers=4, pre_seq_length=10, aft_seq_length=10, train_data_paths='small_2000_10.npz', valid_data_paths='small_2000_10.npz', test_data_paths='small_2000_10.npz'):

    img_width = 256

    train_data_paths = os.path.join(data_root, train_data_paths)
    valid_data_paths = os.path.join(data_root, valid_data_paths)
    test_data_paths = os.path.join(data_root, test_data_paths)

    train_inputs = RadarDataset(train_data_paths,
                                    sample_shape=(pre_seq_length+aft_seq_length, 1, img_width, img_width),    #(20 , 1, 256,256)
                                    input_len=pre_seq_length)                                            
    valid_inputs = RadarDataset(valid_data_paths,
                                   sample_shape=(pre_seq_length+aft_seq_length, 1, img_width, img_width),
                                   input_len=pre_seq_length)
    test_inputs = RadarDataset(test_data_paths,
                                   sample_shape=(pre_seq_length+aft_seq_length, 1, img_width, img_width),
                                   input_len=pre_seq_length)
    
    train_loaders = torch.utils.data.DataLoader(train_inputs, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, pin_memory=True)
    valid_loaders = torch.utils.data.DataLoader(valid_inputs, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loaders = torch.utils.data.DataLoader(test_inputs, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, drop_last=True)
 

    return train_loaders, valid_loaders, test_loaders


if __name__ == '__main__':
    train_loaders, valid_loaders, test_loaders = load_data(batch_size=2, val_batch_size=2, data_root='/home/lfr/mnt/Radar_Data')
    for i, (x, y) in enumerate(train_loaders):
        print(x.shape, y.shape)  # [2, 10, 256, 256, 1], [2, 10, 256, 256, 1]
        break