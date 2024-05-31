import copy

from torch.utils.data import Dataset
import torch
import numpy as np


def normal(traindata):
    original_shape = traindata.size()
    flattened_tensor = traindata.view(traindata.size(0), -1)
    normalized_tensor = torch.nn.functional.normalize(flattened_tensor, dim=1)
    normalized_tensor = normalized_tensor.view(original_shape)
    return normalized_tensor


class mydataset(Dataset):
    def __init__(self, f_pos, t_pos):
        super().__init__()
        self.features = np.load(f_pos)
        self.features = torch.tensor(self.features)
        self.features.to(torch.float32)
        # self.features = normal(self.features)
        # self.features = self.features.permute(1, 2, 0)
        
        self.target = np.load(t_pos)
        self.target = torch.tensor(self.target)
        self.target.to(torch.float32)
        # self.target = normal(self.target)
        # self.target = self.target.permute(1, 2, 0)

    # def backmain(self):
    #     t1 = []
    #     t2 = []
    #     len = self.__len__()
    #     for i in range(0, len - 1):
    #         t2.append(self.features[i])
    #         t2.append(self.target[i])
    #         t1.append(copy.deepcopy(t2))
    #         t2.clear()
    #     return t1

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        data = (self.features[idx], self.target[idx])
        return data