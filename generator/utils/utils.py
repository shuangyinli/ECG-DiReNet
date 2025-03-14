import torch 
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pywt
import time
import os

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, input, target):
        loss = torch.mean((input - target) ** 2).sum(dim=0)
        print("loss", loss)
        return loss


class EMA():
    def __init__(self, betas):
        super(EMA, self).__init__()
        self.betas = betas

    def update_model_param(self, old_model, current_model):
        for old_param, cur_param in zip(old_model.parameters(), current_model.parameters()):
            old_data, cur_data = old_param.data, cur_param.data
            old_param.data = cur_data if old_data is None else old_data * self.betas + (1 - self.betas) * cur_data



# 进行小波变换
def wavelet_transform(ecg_data):
    # 用db5作为小波基，对心电数据进行9尺度小波变换
    coeffs = pywt.wavedec(data=ecg_data, wavelet='sym8', level=8)
    #cA9 , cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    #cA6,cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)
    #cA9.fill(0)
    coeffs = [cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
    # 将其他中低频信号按软阈值公式滤波
    for i in range(0, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='sym8')
    return rdata

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def Normalization2(x):
    x = torch.tensor(x, dtype=torch.float32)  
    mean_val = torch.mean(x)
    max_val = torch.max(x)
    min_val = torch.min(x)
    return [(float(i) - mean_val) / (max_val - min_val) for i in x]




def save_checkpoint(path, epoch,net,  optimizer, dev_best_loss,iter_step,verbose=True):
    data = {
        'epoch': epoch,
        'iter': iter_step,
        'time': time.time(),
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dev_best_loss':dev_best_loss,
    }

    # safe replacement of old checkpoint
    temp_file = None
    if os.path.exists(path):
        temp_file = path + '.old'
        os.rename(path, temp_file)

    # save the new checkpoint
    with open(path, 'wb') as fp:
        torch.save(data, fp)
        fp.flush()
        os.fsync(fp.fileno())

    # remove the old checkpoint
    if temp_file is not None:
        os.unlink(path + '.old')