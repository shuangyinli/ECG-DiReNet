
from typing import List, Any
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import math
class SelfAttention(nn.Module):
    def __init__(self, vector_size, dim, hidden_dim=16, num_heads=4):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.to_qkv = nn.Conv1d(dim, 3*hidden_dim, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.attention = nn.MultiheadAttention(vector_size, num_heads=num_heads, batch_first=True)
        self.normalization = nn.LayerNorm(vector_size)

    def forward(self, x):
        x_norm = self.normalization(x)
        qkv = self.to_qkv(x_norm)
        q = qkv[:,:self.hidden_dim, :]
        k = qkv[:,self.hidden_dim:2*self.hidden_dim, :]
        v = qkv[:,2*self.hidden_dim:, :]
        h, _ = self.attention(q, k, v)
        h = self.to_out(h)
        h = h + x
        return h

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim,  number_of_diffusions,kernel_size = 5, n_heads = None, hidden_dim = None):
        cutted_channels = int((in_channels + out_channels) // 2) + 1
        super(DownsamplingBlock,self).__init__()
        self.kernel_size = kernel_size
        self.time_emb = TimeEmbedding(dim, number_of_diffusions, n_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels,cutted_channels,self.kernel_size,padding="same")
        self.conv2 = nn.Conv1d(cutted_channels,cutted_channels,1,padding="same")
        self.conv3 = nn.Conv1d(cutted_channels,out_channels,self.kernel_size,padding="same")
        self.down = nn.Conv1d(out_channels,out_channels,3,2,padding=1)
        self.layer_norm1 = nn.LayerNorm([cutted_channels,dim])
        self.layer_norm2 = nn.LayerNorm([out_channels,dim])
        self.res_conv = nn.Conv1d(in_channels,out_channels,1) if in_channels!=out_channels else nn.Identity()
        if n_heads is None and hidden_dim is None:
            self.attention = SelfAttention(dim, cutted_channels)
            self.attention2 = SelfAttention(dim, out_channels)
        elif n_heads is None and hidden_dim:
            self.attention = SelfAttention(dim, cutted_channels, hidden_dim=hidden_dim)
            self.attention2 = SelfAttention(dim, out_channels, hidden_dim=hidden_dim)
        elif n_heads and hidden_dim is None:
            self.attention = SelfAttention(dim, cutted_channels, num_heads=n_heads)
            self.attention2 = SelfAttention(dim, out_channels, num_heads=n_heads)
        elif n_heads and hidden_dim:
            self.attention = SelfAttention(dim, cutted_channels, hidden_dim=hidden_dim, num_heads=n_heads)
            self.attention2 = SelfAttention(dim, out_channels, hidden_dim=hidden_dim, num_heads=n_heads)

    def forward(self, x,t, h=None):
        initial_x = x
        t = self.time_emb(t) #.repeat(1,1, x.shape[2])
        x = x + t

        shortcut = self.conv1(x)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.conv2(shortcut)
        shortcut = self.attention(shortcut)
        out = self.conv3(shortcut)
        out = self.layer_norm2(out)
        h = out
        out = self.down(out)
        out = F.mish(out)

        return h, out

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dim, number_of_diffusions,
                 kernel_size=5, up_dim=None, n_heads=None, hidden_dim=None):
        n_shortcut = int((n_inputs + n_outputs) // 2)
        super(UpsamplingBlock, self).__init__()
        self.kernel_size = kernel_size

        # CONV 1
        self.pre_shortcut_convs = nn.Conv1d(n_inputs*2, n_shortcut, self.kernel_size, padding="same")
        self.shortcut_convs = nn.Conv1d(n_shortcut, n_shortcut, self.kernel_size, padding="same")
        self.post_shortcut_convs = nn.Conv1d(n_shortcut, n_outputs, self.kernel_size, padding="same")
        if up_dim is None:
            self.up = nn.ConvTranspose1d(n_inputs, n_inputs, kernel_size=4, stride=2, padding=1)  # padding=1
        else:
            self.up = nn.ConvTranspose1d(up_dim, up_dim, kernel_size=4, stride=2, padding=1)
        self.layer_norm1 = nn.LayerNorm(2*dim)
        self.layer_norm2 = nn.LayerNorm(2*dim)
        self.res_conv = nn.Conv1d(2*n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        if n_heads is None and hidden_dim is None:
            self.attention = SelfAttention(2*dim, n_shortcut)
            self.attention2 = SelfAttention(2 * dim, n_outputs)
        elif n_heads is None and hidden_dim:
            self.attention = SelfAttention(2*dim, n_shortcut, hidden_dim=hidden_dim)
            self.attention2 = SelfAttention(2 * dim, n_outputs, hidden_dim=hidden_dim)
        elif n_heads and hidden_dim is None:
            self.attention = SelfAttention(2*dim, n_shortcut, num_heads=n_heads)
            self.attention2 = SelfAttention(2 * dim, n_outputs, num_heads=n_heads)
        elif n_heads and hidden_dim:
            self.attention = SelfAttention(2*dim, n_shortcut, hidden_dim=hidden_dim, num_heads=n_heads)
            self.attention2 = SelfAttention(2 * dim, n_outputs, hidden_dim=hidden_dim, num_heads=n_heads)

        self.time_emb = TimeEmbedding(2*dim, number_of_diffusions, n_channels=n_inputs)

    def forward(self, x, h, t):
        x = self.up(x)
        initial_x = x
        t = self.time_emb(t)  # .repeat(1, 1, x.shape[2])
        x = x + t
        if h is None:
            h = x
        shortcut = torch.cat([x, h], dim=1)
        shortcut = self.pre_shortcut_convs(shortcut)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)
        shortcut = self.attention(shortcut)
        out = self.post_shortcut_convs(shortcut)
        out = self.layer_norm2(out)
        out = F.mish(out)
        a = self.res_conv(torch.cat([initial_x, h], dim=1))
        out = out + a
        return out


 
class ECGunetChannels(nn.Module):
    def __init__(self, number_of_diffusions, kernel_size=3, num_levels=3, n_channels=1, resolution=2048): 
        super(ECGunetChannels, self).__init__()

        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.number_of_diffusions = number_of_diffusions 

        # Only odd filter kernels allowed
        assert (kernel_size % 2 == 1)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        self.downsampling_blocks.append(
            DownsamplingBlock(in_channels=n_channels, out_channels=2 * n_channels,
                              dim=resolution, number_of_diffusions=number_of_diffusions,kernel_size=kernel_size + 2, n_heads=8, hidden_dim=8))
        self.downsampling_blocks.append(
            DownsamplingBlock(in_channels=2 * n_channels, out_channels=4 * n_channels,
                              dim=resolution // 2, number_of_diffusions=number_of_diffusions,kernel_size=kernel_size))
        current_resolution = resolution//2
        for i in range(2, self.num_levels - 1):
            current_resolution = current_resolution >> i
            self.downsampling_blocks.append(
                DownsamplingBlock(in_channels=n_channels * 2 ** i, out_channels=n_channels * 2 ** (i + 1),
                                  dim=current_resolution,number_of_diffusions=number_of_diffusions, kernel_size=kernel_size))

        self.upsampling_blocks.append(
            UpsamplingBlock(n_inputs=n_channels * 2 ** (num_levels-1), n_outputs=n_channels * 2 ** (num_levels - 2),
                            dim=current_resolution // 2, number_of_diffusions=number_of_diffusions,
                            kernel_size=kernel_size))

        for i in reversed(range(1, self.num_levels - 2)):
            current_resolution = resolution >> (i + 1)
            self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=n_channels * 2 ** (i + 2), n_outputs=n_channels * 2 ** i,
                                dim=current_resolution, number_of_diffusions=number_of_diffusions,
                                kernel_size=kernel_size))
        current_resolution = resolution // 2
        self.upsampling_blocks.append(
            UpsamplingBlock(n_inputs=n_channels * 2 ** (num_levels - 2), n_outputs=n_channels,
                            dim=current_resolution, number_of_diffusions=number_of_diffusions, kernel_size=kernel_size,
                            n_heads=8))  # , up_dim=2))

        self.time_emb = TimeEmbedding(resolution >> (self.num_levels - 1), number_of_diffusions,
                                      n_channels=n_channels * 2 ** (self.num_levels - 1))

        self.bottleneck_conv1 = nn.Conv1d(n_channels * 2 ** (self.num_levels - 1),
                                          n_channels * 2 ** (self.num_levels - 1), kernel_size=3, padding="same")
        #self.bottleneck_conv1_2 = nn.Conv1d(n_channels * 2 ** (self.num_levels - 1),
        #                                    n_channels * 2 ** (self.num_levels -1), kernel_size=3, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(n_channels * 2 ** (self.num_levels-1),
                                          n_channels * 2 ** (self.num_levels - 1), kernel_size=3, padding="same")
        self.attention_block = SelfAttention(resolution >> (self.num_levels-1),
                                             n_channels * 2 ** (self.num_levels-1), hidden_dim=32)
        self.bottleneck_layer_norm1 = nn.LayerNorm(resolution >> (self.num_levels - 1))
        self.bottleneck_layer_norm2 = nn.LayerNorm(resolution >> (self.num_levels - 1))

        self.output_conv = nn.Sequential(nn.Conv1d( n_channels, n_channels, 3, padding="same"), nn.Mish(),
                                         nn.Conv1d(n_channels, n_channels, 1, padding="same"))

    def forward(self, x, t,label):
        device = x.device
        shortcuts = []
        labels = label.unsqueeze(1)
        labels = torch.broadcast_to(labels,x.shape).to(device)
        x = torch.add(x, labels)
        out = x
        for block in self.downsampling_blocks:
            h, out = block(out,t)
            shortcuts.append(h)

        del shortcuts[-1]
        old_out = out
        tt = self.time_emb(t)  # [:,None,:].repeat(1, out.shape[1], 1)
        out = out + tt
        out = self.bottleneck_conv1(out)
        out = self.bottleneck_layer_norm1(out)
        out = F.mish(out)
        self_attention1 = self.attention_block(out)
        out = self.bottleneck_conv2(self_attention1)
        out = self.bottleneck_layer_norm2(out)
        out = F.mish(out) + old_out / math.sqrt(2)  # residiual connection normalization
        out = self.upsampling_blocks[0](out, None, t)

        for idx, block in enumerate(self.upsampling_blocks[1:]):
            out = block(out, shortcuts[-1 - idx], t)

        out = self.output_conv(out)
        return out




class TimeEmbedding(nn.Module):
    def __init__(self, dim, number_of_diffusions, n_channels=1, dim_embed=64, dim_latent=128):
        super(TimeEmbedding, self).__init__()
        self.dim_latent = dim_latent
        self.number_of_diffusions = number_of_diffusions
        self.n_channels = n_channels
        self.fc1 = nn.Linear(dim_embed, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim)
        if n_channels >= 1:
            self.out_conv = nn.Conv1d(1, n_channels, 1)
        self.dim_embed = dim_embed
        self.embeddings = nn.Parameter(self.embed_table())
        self.embeddings.requires_grad = False

    def embed_table(self):
        t = torch.arange(self.number_of_diffusions) + 1
        half_dim = self.dim_embed // 2
        emb = 10.0 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :] 
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, t):
        emb = self.embeddings[t, :]  
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        if self.n_channels >= 1:
            out = out.unsqueeze(1)
            out = self.out_conv(out) 
        return out

 

class ECGLDM(nn.Module):
    def __init__(self, time_steps=1000, n_channels=1, device='cpu'):
        super(ECGLDM, self).__init__()
        self.time_steps = time_steps
        self.channels_num = n_channels
        self.betas = nn.Parameter(torch.linspace(1e-4, 0.02, time_steps), requires_grad=False).to(device)
        self.alpha = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar = 1 - self.alpha_bar
        self.one_minus_alpha_bar_sqrt = torch.sqrt(self.one_minus_alpha_bar)

    def cosine_beta(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, steps, steps)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, min=0.00001, max=0.999)
        return betas

    def forward_diffuse(self, x0, t, noise=None):
        device = x0.device  
        if noise is None:
            noise = torch.normal(0, 1, size=[t.shape[0], self.channels_num, x0.shape[-1]], device=device)
        alpha_bar_sqrt_t = self.alpha_bar_sqrt[t].to(device)  # 1D
        alpha_bar_sqrt_t = alpha_bar_sqrt_t.repeat(x0.shape[-1], self.channels_num, 1).transpose(0, 2)
        one_minus_alpha_bar_t_sqrt = self.one_minus_alpha_bar_sqrt[t].to(device)
        one_minus_alpha_bar_t_sqrt = one_minus_alpha_bar_t_sqrt.repeat(x0.shape[-1], self.channels_num, 1).transpose(0, 2)

        x_t = alpha_bar_sqrt_t * x0 + one_minus_alpha_bar_t_sqrt * noise
        return x_t, noise

    def posterior_distribution(self, net, xt, t):
        net.eval()
        device = xt.device  
        noise_predict = net(xt, t).to(device)

        coeff = (1 / torch.sqrt(self.alpha[t])).repeat(xt.shape[2], 1).transpose(0, 1).to(device)
        coeff = torch.unsqueeze(coeff, 1)
        coeff = coeff.repeat(1, self.channels_num, 1)

        beta_t = self.betas[t].repeat(xt.shape[2], 1).transpose(0, 1).to(device)
        beta_t = torch.unsqueeze(beta_t, 1)
        beta_t = beta_t.repeat(1, self.channels_num, 1)

        one_minus_alpha_bar_t_sqrt = self.one_minus_alpha_bar_sqrt[t].repeat(xt.shape[2], 1).transpose(0, 1).to(device)
        one_minus_alpha_bar_t_sqrt = torch.unsqueeze(one_minus_alpha_bar_t_sqrt, 1)
        one_minus_alpha_bar_t_sqrt = one_minus_alpha_bar_t_sqrt.repeat(1, self.channels_num, 1)

        coeff_noise = beta_t / one_minus_alpha_bar_t_sqrt
        post_mean = coeff * (xt - coeff_noise * noise_predict)

        return post_mean

    def p_sample(self, net, x_t, t):
        net.eval()
        device = x_t.device  
        sigma_up = 1 - self.alpha_bar[t - 1] * self.betas[t]
        sigma_up = sigma_up.repeat(x_t.shape[2], 1).transpose(0, 1).to(device)
        sigma_down = 1 - self.alpha_bar[t]
        sigma_down = sigma_down.repeat(x_t.shape[2], 1).transpose(0, 1).to(device)
        sigma_t = sigma_up / sigma_down
        sigma_t = torch.unsqueeze(sigma_t, 1)
        sigma_t = sigma_t.repeat(1, self.channels_num, 1)

        z = torch.randn_like(x_t, device=device)
        p_mean = self.posterior_distribution(net, x_t, t)

        sample = p_mean + sigma_t * z
        return sample

    def p_sample_loop(self, net, time_steps, shape):
        cur_x = torch.randn(shape)
        x_seq = []
        for i in reversed(range(0, time_steps)):
            t = i * torch.ones(shape[0], dtype=torch.long, device=cur_x.device)
            with torch.no_grad():
                cur_x = self.p_sample(net, cur_x, t)
                x_seq.append(cur_x)
        return x_seq

    def con_fn(self, x, t, classifier, y=None):
        assert y is not None
        device = x.device  
        with torch.enable_grad():
            x_temp = x.squeeze(1).detach().to(device).requires_grad_(True)
            logits = classifier(x_temp).to(device)
            coeff = self.one_minus_alpha_bar_sqrt[t].unsqueeze(1).repeat(x.shape[1], 1).to(device)
            log_probs = F.log_softmax(logits, dim=-1).to(device)
            selected = log_probs[range(len(logits)), y.view(-1).type(torch.long)]
            gradient = torch.autograd.grad(selected, x_temp, grad_outputs=torch.ones_like(selected))[0]
            return gradient * coeff
