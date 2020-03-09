import torch
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor_data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor_data = data.detach()
    if cuda:
        tensor_data = tensor_data.cuda()
    return tensor_data


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()