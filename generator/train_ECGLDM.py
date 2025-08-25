from torch.utils.data import TensorDataset
import os.path
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from model.ECGLDM import ECGLDM,ECGunetChannels
from utils.utils import EMA ,save_checkpoint,load_and_combine_npz
from argparse import ArgumentParser
from tqdm import trange,tqdm
from copy import deepcopy

def train(dataloader, net, diffusion_model, optimizer, loss_func, schedule, device, iter_step, classifier=None):
    loss_list = []
    net.train()
    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for ecg, label in pbar:
            optimizer.zero_grad()
            ecg, label = ecg.to(device), label.to(device)
            t = torch.randint(1, diffusion_model.time_steps - 1, size=(ecg.shape[0],)).to(device)
            xt, noise = diffusion_model.forward_diffuse(ecg, t)
            noise = noise.to(device)
            xt = xt.to(device)

            noise_pred = net(xt, t, label).to(device)

            loss = loss_func(noise_pred, noise)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            schedule.step()
            iter_step = iter_step + 1
            pbar.set_postfix(loss=loss.item())
    return sum(loss_list) / len(loss_list), iter_step

def dev(dataloader, net, diffusion_model, optimizer, loss_func, schedule, device, classifier=None):
    loss_list = []
    with torch.no_grad():
        net.eval()
        with tqdm(dataloader, desc="Validation", unit="batch") as pbar:
            for ecg, label in pbar:
                t = torch.randint(1, diffusion_model.time_steps - 1, size=(ecg.shape[0],))
                t = t.to(device)
                ecg, label = ecg.to(device), label.to(device)
                xt, noise = diffusion_model.forward_diffuse(ecg, t)
                noise = noise.to(device)
                xt = xt.to(device)
                noise_pred = net(xt, t, label)
                loss = loss_func(noise_pred, noise)
                loss_list.append(loss.item())
                pbar.set_postfix(loss=loss.item())

    return sum(loss_list) / len(loss_list)



def warmup_lr(step):
    return min(step, 1600) / 1600

if __name__ == "__main__":
    parser = ArgumentParser(description='ECG Diffusion Model Training')
    parser.add_argument('--train_data_path', type=str, default="../data/train.npz")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/checkpoint.pth")
    parser.add_argument('--best_model_path', type=str, default="./weight/best_model.pth")
    parser.add_argument('--channels_num', type=int, default=1)
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoches', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    ckpt_dir = os.path.dirname(args.checkpoint_path)
    if ckpt_dir and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    weight_dir = os.path.dirname(args.best_model_path)
    if weight_dir and not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    diffuse_model = ECGLDM(n_channels=args.channels_num, device=device)
    ecg_net = ECGunetChannels(diffuse_model.time_steps, kernel_size=5, n_channels=args.channels_num,resolution=args.dim).to(device)

    ema = EMA(betas=0.9999)
    start_epoch = 0
    iter_step = 0
    optimizer = torch.optim.Adam(ecg_net.parameters(), lr=args.lr)
    loss_fuc = nn.MSELoss()

    train_data, train_label, _ = load_and_combine_npz(
        file_paths=[args.train_data_path],
        file_specific_limit={},
        shuffle=True,
        include_csv_names=False
    )
    train_data = train_data.unsqueeze(1)
    train_label = train_label.unsqueeze(1)

    idx = np.arange(train_data.shape[0])
    train_idx, dev_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True,
                                          stratify=train_label.cpu().numpy() if train_label.ndim == 1 else None)

    trainData = train_data[train_idx]
    devData = train_data[dev_idx]
    trainLabel = train_label[train_idx]
    devLabel = train_label[dev_idx]

    train_dataset = TensorDataset(trainData, trainLabel)
    dev_dataset = TensorDataset(devData, devLabel)

    train_DataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=8, pin_memory=True)
    dev_DataLoader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)

    if args.resume and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        start_epoch = checkpoint.get('epoch', 0)
        iter_step = checkpoint.get('iter', 0)
        net_dict = ecg_net.state_dict()
        pred = checkpoint['net']
        state_dict = {k: v for k, v in pred.items() if k in net_dict}
        net_dict.update(state_dict)
        ecg_net.load_state_dict(net_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        dev_best_loss = checkpoint.get('dev_best_loss', 100.0)
 
    else:
        dev_best_loss = 100.0

    for g in optimizer.param_groups:
        if 'initial_lr' not in g:
            g['initial_lr'] = g['lr']
            
    if args.resume and os.path.exists(args.checkpoint_path):
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr, last_epoch=iter_step - 1)
    else:
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr, last_epoch=-1)

    for i in trange(start_epoch + 1, args.epoches):
        torch.cuda.empty_cache()
        cpy_net = deepcopy(ecg_net)
        loss, iter_step = train(dataloader=train_DataLoader, net=ecg_net, diffusion_model=diffuse_model,
                                optimizer=optimizer, loss_func=loss_fuc, schedule=schedule, device=device,
                                iter_step=iter_step)
        torch.cuda.empty_cache()
        dev_loss = dev(dataloader=dev_DataLoader, net=ecg_net, diffusion_model=diffuse_model,
                       optimizer=optimizer, loss_func=loss_fuc, schedule=schedule, device=device)

        ema.update_model_param(cpy_net, ecg_net)

        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(ecg_net.state_dict(), args.best_model_path)
        save_checkpoint(args.checkpoint_path, i, ecg_net, optimizer, dev_best_loss, iter_step)
        print(f'Epoch: {i + 1:04d}, training loss={loss:.8f}')
        print(f'Epoch: {i + 1:04d}, dev loss={dev_loss:.8f}')