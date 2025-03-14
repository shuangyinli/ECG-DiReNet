import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import EMA,MSE,wavelet_transform,Normalization2
from model.ECGLDM import ECGunetChannels
from argparse import ArgumentParser
from tqdm import tqdm
import os
class DiffusionModel(nn.Module):
    def __init__(self, time_steps=1000, n_channels=1, device="cpu"):
        super(DiffusionModel, self).__init__()
        self.time_steps = time_steps  # Number of diffusion steps
        self.number_of_channels = n_channels  # Number of leads
        self.betta = torch.linspace(1e-4, 0.02, time_steps, device=device)  # β
        # self.betta = self.cosine_beta_schedule(time_steps) # It's a list
        self.alpha = 1 - self.betta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)  # Cumulative product
        self.alpha_cumprod_sqrt = torch.sqrt(self.alpha_cumprod)  # Square root of cumulative product
        self.alpha_cumprod_sqrt_recip = 1 / torch.sqrt(self.alpha_cumprod)  # Reciprocal of square root of cumulative product
        self.alpha_cumprod_sqrt_recip_minus_one = torch.sqrt(1 / self.alpha_cumprod - 1)  # Square root of reciprocal of (cumulative product - 1)
        self.one_minus_alpha_cumprod = 1 - self.alpha_cumprod  # 1 - cumulative product
        self.one_minus_alpha_cumprod_sqrt = torch.sqrt(self.one_minus_alpha_cumprod)  # Square root of (1 - cumulative product)
        self.alphas_cumprod_prev = np.append(1., self.alpha_cumprod.cpu().numpy()[:-1])
        self.posterior_variance = self.betta.cpu().numpy() * (1. - self.alphas_cumprod_prev) / (1. - self.alpha_cumprod.cpu().numpy())
        self.posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(self.posterior_variance, 1e-20)), dtype=torch.float32, device=device)


    def cosine_beta(self, timesteps, s=0.008):
        """
           Cosine schedule
           As proposed in https://openreview.net/forum?id=-NEXDKk8gZ 
        """
        steps = timesteps + 1
        x = torch.linspace(0, steps, steps)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, min=0.00001, max=0.999)
        return betas

    @staticmethod
    def _extract(a, t, x_shape):
        bs, = t.shape
        assert x_shape[0] == bs
        # Move the tensor to the same device as x
        out = torch.gather(a, 0, t).to(t.device)
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def forward_diffuse(self, x0, t, noise=None):
        assert x0.shape[0] == t.shape[0]
        device = x0.device  # Get the device of input x0
        if noise is None:
            noise = torch.normal(0, 1, size=[t.shape[0], self.number_of_channels, x0.shape[-1]], device=device)
        alpha_bar_sqrt_t = self.alpha_cumprod_sqrt[t].to(device)  # 1D
        alpha_bar_sqrt_t = alpha_bar_sqrt_t.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0, 2)
        one_minus_alpha_bar_t_sqrt = self.one_minus_alpha_cumprod_sqrt[t].to(device)
        one_minus_alpha_bar_t_sqrt = one_minus_alpha_bar_t_sqrt.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0, 2)

        x_t = alpha_bar_sqrt_t * x0 + one_minus_alpha_bar_t_sqrt * noise
        return x_t, noise

    def posterior_distribution(self, net, xt, t, classifier=None):
        device = xt.device  # Get the device of input xt
        net.eval()
        noise_predict = net(xt, t).to(device)

        coeff = 1 / torch.sqrt(self.alpha[t])
        coeff = coeff.repeat(xt.shape[2], 1).transpose(0, 1)
        coeff = torch.unsqueeze(coeff, 1)
        coeff = coeff.repeat(1, self.number_of_channels, 1).to(device)

        beta_t = self.betta[t]
        beta_t = beta_t.repeat(xt.shape[2], 1).transpose(0, 1)
        beta_t = torch.unsqueeze(beta_t, 1)
        beta_t = beta_t.repeat(1, self.number_of_channels, 1).to(device)

        one_minus_alpha_bar_t_sqrt = self.one_minus_alpha_cumprod_sqrt[t]
        one_minus_alpha_bar_t_sqrt = one_minus_alpha_bar_t_sqrt.repeat(xt.shape[2], 1).transpose(0, 1)
        one_minus_alpha_bar_t_sqrt = torch.unsqueeze(one_minus_alpha_bar_t_sqrt, 1)
        one_minus_alpha_bar_t_sqrt = one_minus_alpha_bar_t_sqrt.repeat(1, self.number_of_channels, 1).to(device)

        coeff_noise = (beta_t / one_minus_alpha_bar_t_sqrt).to(device)
        part1 = (coeff_noise * noise_predict).to(device)
        part2 = (xt - part1).to(device)
        part3 = (coeff * part2).to(device)

        return part3

    def mean_posterior(self, x0, t, xt=None):
        """
        x0 - points
        t - list of time, int
        """
        device = x0.device  # Get the device of input x0
        alpha_cum_prod_sqrt_t_1 = self.alpha_cumprod_sqrt[t - 1].to(device)
        alpha_sqrt_t = torch.sqrt(self.alpha[t]).to(device)
        one_minus_alpha_cumprod_t_1 = self.one_minus_alpha_cumprod[t - 1].to(device)
        one_minus_alpha_cumprod_t = self.one_minus_alpha_cumprod[t].to(device)

        alpha_cum_prod_sqrt_t_1 = alpha_cum_prod_sqrt_t_1.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0, 2)
        alpha_sqrt_t = alpha_sqrt_t.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0, 2)
        one_minus_alpha_cumprod_t_1 = one_minus_alpha_cumprod_t_1.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0, 2)
        one_minus_alpha_cumprod_t = one_minus_alpha_cumprod_t.repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0, 2)
        betta_t = self.betta[t].repeat(x0.shape[-1], self.number_of_channels, 1).transpose(0, 2).to(device)

        if xt is None:
            xt, _ = self.forward_diffuse(x0, t)
        part1 = alpha_cum_prod_sqrt_t_1 * betta_t * x0
        part2 = alpha_sqrt_t * one_minus_alpha_cumprod_t_1 * xt
        mu = (part1 + part2) / one_minus_alpha_cumprod_t
        return xt, mu

    def predict_start_from_noise(self, x_t, t, noise):
        device = x_t.device  # Get the device of input x_t
        coeff_xt = self.alpha_cumprod_sqrt_recip[t]
        coeff_xt = coeff_xt.repeat(x_t.shape[2], 1).transpose(0, 1)
        coeff_xt = coeff_xt.to(device)
        coeff_xt = coeff_xt.unsqueeze(1)

        coeff_noise = self.alpha_cumprod_sqrt_recip_minus_one[t]
        coeff_noise = coeff_noise.repeat(x_t.shape[2], 1).transpose(0, 1)
        coeff_noise = coeff_noise.to(device)
        coeff_noise = coeff_noise.unsqueeze(1)

        x0 = coeff_xt * x_t - coeff_noise * noise
        return x0

    def backward_pass(self, input):
        k_steps = 1000
        device = input.device  # Get the device of input
        xi = torch.randn_like(input).to(device)
        for i in reversed(range(1, k_steps)):
            t = i * torch.ones(input.shape[0], dtype=torch.long).to(device)
            _, mu = self.mean_posterior(input, t, xi)
            delta = mu - xi
            if i % 200 == 0:
                print(i, delta.abs().mean())
            xi = mu
        plt.plot(xi[0, :].cpu().numpy())

    def generation_from_net(self, net, number_of_points, dim=512):
        net.eval()
        device = next(net.parameters()).device  # Get the device of the network
        xi = torch.randn(number_of_points, self.number_of_channels, dim).to(device)
        for i in reversed(range(1, self.time_steps)):
            t = i * torch.ones(number_of_points, dtype=torch.long).to(device)
            with torch.no_grad():
                noise = torch.sqrt(self.betta[i]) * torch.randn(number_of_points, self.number_of_channels, dim).to(device)
                delta_mu = net(xi, t)
                xi = xi + delta_mu + noise
        xi = xi.cpu()
        return xi
    def generation_from_net_epsilon(self, net, label, number_of_points, dim=256):
        net.eval()
        device = next(net.parameters()).device  # Get the device of the network
        xi = torch.randn(number_of_points, dim).unsqueeze(1).to(device)
        pbar = tqdm(reversed(range(1, self.time_steps)), desc="Generating time_steps (0/%d)" % self.time_steps)
        for i in pbar:
            with torch.no_grad():
                t = i * torch.ones(number_of_points, dtype=torch.long).to(device)
                noise = torch.sqrt(self.betta[i]) * torch.randn(number_of_points, dim).unsqueeze(1).to(device)
                noise1 = torch.randn(number_of_points, dim).unsqueeze(1).to(device)

                sigma_up = (1 - self.alpha_cumprod[t - 1]) * self.betta[t]
                sigma_up = sigma_up.repeat(xi.shape[2], 1).transpose(0, 1).to(device)
                sigma_down = 1 - self.alpha_cumprod[t]
                sigma_down = sigma_down.repeat(xi.shape[2], 1).transpose(0, 1).to(device)
                sigma_t = sigma_up / sigma_down
                sigma_t = torch.unsqueeze(sigma_t, 1).repeat(1, self.number_of_channels, 1).to(device)
                labels = torch.zeros(xi.shape[0]) + label
                labels = labels.to(device)
                epsilon = net(xi, t, labels.unsqueeze(1))
                x0 = self.predict_start_from_noise(xi, t, epsilon)
                _, mu = self.mean_posterior(x0, t, xi)
                mu = mu.to(device)
                result = (t == 0)
                temp = result.to(torch.float32)
                nonzero_mask = torch.reshape(1 - temp, [xi.shape[0]] + [1] * (len(xi.shape) - 1)).to(device)
                model_log_variance = self._extract(self.posterior_log_variance_clipped.to(device), t, xi.shape).to(device)
                xi = mu + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise1
                xi = xi.clamp(min=-1.0, max=1.0)
            pbar.set_description("Generating time_steps (%d/%d)" % (i, self.time_steps))
        pbar.close()
        return xi
    
    def con_fn(self, x, t, classifier, y=None):
        assert y is not None
        device = x.device  # Get the device of input x
        with torch.enable_grad():
            x_temp = x.squeeze(1).detach().to(device).requires_grad_(True)
            logits = classifier(x_temp).to(device)
            coeff = self.one_minus_alpha_cumprod_sqrt[t].unsqueeze(1).repeat(x.shape[1], 1).to(device)
            log_probs = F.log_softmax(logits, dim=-1).to(device)
            selected = log_probs[range(len(logits)), y.view(-1).type(torch.long)]
            gradient = torch.autograd.grad(selected, x_temp, grad_outputs=torch.ones_like(selected))[0]
            return gradient * coeff


if __name__ == "__main__":
    # Create an argument parser
    parser = ArgumentParser(description='Diffusion Model Generation')
    # Add arguments
    parser.add_argument('--channels_num', type=int, default=1, help='Number of channels')
    parser.add_argument('--save_path', type=str, default="./generated/generated_data.npz", help='Path to save the generated data')
    parser.add_argument('--checkpoint_path', type=str,
                        default="./weight/bestUnet_Channel_2_step_forwardwithLabel_warmup1600_lr00005.pth",
                        help='Path to the checkpoint file')
    parser.add_argument('--generate_num', type=int, default=100, help='Number of samples to generate each iteration')
    parser.add_argument('--iterations', type=int, default=2, help='Number of iterations')
    parser.add_argument('--label', type=int, default=0, help='ECG Label')

    parser.add_argument('--dim', type=int, default=2048, help='Dimension of the generated data')
    args = parser.parse_args()

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
        print(f"Created weight directory: {save_path_dir}")

    # Initialize the models
    diffuse_model = DiffusionModel(n_channels=args.channels_num,device=device)
    ecg_net = ECGunetChannels(diffuse_model.time_steps, kernel_size=5, n_channels=args.channels_num).to(device)
    # Load the model weights
    checkpoint = torch.load(args.checkpoint_path)
    ecg_net.load_state_dict(checkpoint)

    fake_list = []
    for i in tqdm(range(args.iterations), desc="Generating data"):
        torch.cuda.empty_cache()
        # Generate data using the diffusion model
        x0 = diffuse_model.generation_from_net_epsilon(ecg_net,  args.label, args.generate_num, dim=args.dim).squeeze(1)
        fake_list.extend(x0.cpu().tolist())

    data = np.array(fake_list)
    
    normalized_data = np.array([Normalization2(x) for x in data])
    transformed_data = np.array([wavelet_transform(x) for x in normalized_data])
    labels = np.zeros(data.shape[0]) + args.label

    print(f"data.shape: {data.shape}")

    np.savez(args.save_path, data=data, label=labels)
    print(f"Data saved to {args.save_path}")
    print(data.shape)