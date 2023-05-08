from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataSet(Dataset):
    def __init__(self, root_x, root_y):
        data_list = os.listdir(root_x)
        self.x_data_path_list = [os.path.join(root_x, str(i)) for i in data_list]
        self.y_data_path_list = [os.path.join(root_y, str(i)) for i in data_list]

    def __getitem__(self, index):
        x_index_path = self.x_data_path_list[index]
        y_index_path = self.y_data_path_list[index]
        x_image = torch.from_numpy(np.array(Image.open(x_index_path)))
        y_image = torch.from_numpy(np.array(Image.open(y_index_path)))

        # 由于这里读取的是灰度图，所以要加一个维度
        x_image = x_image.unsqueeze(0)
        y_image = y_image.unsqueeze(0)

        return x_image, y_image

    def __len__(self):
        return len(self.x_data_path_list)