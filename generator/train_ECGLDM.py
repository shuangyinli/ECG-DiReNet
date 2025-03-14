
from torch.utils.data import TensorDataset
import os.path
from tqdm import trange
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import   DataLoader
from sklearn.model_selection import train_test_split
from model.ECGLDM import ECGLDM,ECGunetChannels
from utils.utils import EMA ,save_checkpoint
from argparse import ArgumentParser
from tqdm import tqdm
def train(dataloader, net, diffusion_model, optimizer, loss_func, schedule, device, iter_step, classifier=None):
    # List to store the loss values for each batch
    loss_list = []
    # Set the network to training mode
    net.train()
    # Wrap the dataloader with tqdm to add a progress bar
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
    # Calculate the average loss over all batches
    return sum(loss_list) / len(loss_list), iter_step

def dev(dataloader, net, diffusion_model, optimizer, loss_func, schedule, device, classifier=None):
    # List to store the loss values for each batch during validation
    loss_list = []
    # Disable gradient calculation since we are in evaluation mode
    with torch.no_grad():
        # Set the network to evaluation mode
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
    # Create an argument parser
    parser = ArgumentParser(description='ECG Diffusion Model Training')
    # Add arguments
    parser.add_argument('--good_data_path', type=str, default="good_data.npz", help='Path to good data npz file')
    parser.add_argument('--bad_data_path', type=str, default="bad_data.npz", help='Path to bad data npz file')
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/Channel_2_step_forwardwithLabel_warmup1600_lr00005.pth",
                        help='Path to the checkpoint file')
    parser.add_argument('--best_unet_path', type=str,
                        default="./weight/bestUnet_Channel_2_step_forwardwithLabel_warmup1600_lr00005.pth",
                        help='Path to save the best UNet model')
    parser.add_argument('--channels_num', type=int, default=1, help='Number of channels')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epoches', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    # Create the checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    # Create the weight directory if it doesn't exist
    weight_dir = os.path.dirname(args.best_unet_path)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        print(f"Created weight directory: {weight_dir}")

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize diffusion model and network
    diffuse_model = ECGLDM(n_channels=args.channels_num,device=device).to(device)
    ecg_net = ECGunetChannels(diffuse_model.time_steps, kernel_size=5, n_channels=args.channels_num).to(device)

    # Initialize EMA, optimizer, and loss function
    ema = EMA(betas=0.9999)
    start_epoch = 0
    iter_step = 0
    optimizer = torch.optim.Adam(ecg_net.parameters(), lr=args.lr)
    loss_fuc = nn.MSELoss()

    # Load npz files
    good_data_npz = np.load(args.good_data_path)
    good_data = good_data_npz['data'].astype(np.float32)
    good_labels = good_data_npz['label'].astype('int')[:, np.newaxis]

    bad_data_npz = np.load(args.bad_data_path)
    bad_data = bad_data_npz['data'].astype(np.float32)
    bad_labels = bad_data_npz['label'].astype('int')[:, np.newaxis]

    # Concatenate data and labels
    datasets = np.concatenate([good_data, bad_data], axis=0)
    datasets = torch.FloatTensor(datasets).unsqueeze(1)
    labels = np.concatenate([good_labels, bad_labels])


    # Split data into training and validation sets
    X_train, X_dev, y_train, y_dev = train_test_split(datasets, labels, test_size=0.2, train_size=0.8,
                                                      random_state=42)
    trainData = torch.Tensor(X_train)
    devData = torch.Tensor(X_dev)
    trainLabel = torch.Tensor(y_train)
    devLabel = torch.Tensor(y_dev)
    train_dataset = TensorDataset(trainData, trainLabel)
    dev_dataset = TensorDataset(devData, devLabel)
    train_DataLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    dev_DataLoader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size)

    # Resume training
    if args.resume and os.path.exists(args.checkpoint_path):
        path_checkpoint = args.checkpoint_path
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        iter_step = checkpoint['iter']
        net_dict = ecg_net.state_dict()
        predictted_net = checkpoint['net']
        state_dict = {k: v for k, v in predictted_net.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        ecg_net.load_state_dict(net_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        dev_best_loss = checkpoint['dev_best_loss']
    else:
        dev_best_loss = 100

    # Learning rate scheduler
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr, last_epoch=start_epoch)

    print("start_epoch:", start_epoch)

    # Training loop
    for i in trange(start_epoch + 1, args.epoches):
        torch.cuda.empty_cache()
        cpy_net = copy.deepcopy(ecg_net)  # Backup the previous ECG network
        loss, iter_step = train(dataloader=train_DataLoader, net=ecg_net, diffusion_model=diffuse_model,
                                optimizer=optimizer, loss_func=loss_fuc, schedule=schedule, device=device,
                                iter_step=iter_step)
        torch.cuda.empty_cache()
        dev_loss = dev(dataloader=dev_DataLoader, net=ecg_net, diffusion_model=diffuse_model,
                       optimizer=optimizer, loss_func=loss_fuc, schedule=schedule, device=device)

        ema.update_model_param(cpy_net, ecg_net)

        if dev_loss < dev_best_loss:
            print("saving model...")
            dev_best_loss = dev_loss
            torch.save(ecg_net.state_dict(), args.best_unet_path)
        save_checkpoint(args.checkpoint_path, i, ecg_net, optimizer, dev_best_loss, iter_step)
        print('Epoch: %04d, training loss=%.8f' % (i + 1, loss))
        print('Epoch: %04d, dev loss=%.8f' % (i + 1, dev_loss))

