
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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
        emb = t[:, None] * emb[None, :]  # 扩展维度，none的那一维什么都没有
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # print("emb.shape:",emb.shape)   # [num_diffusions x dim_emb]
        # print("emb:", emb)
        return emb

    def forward(self, t):
        emb = self.embeddings[t, :]  
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        if self.n_channels >= 1:
            out = out.unsqueeze(1)
            out = self.out_conv(out)  # 12导联 每个导联都是经历相同的时间
        return out


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
        #self.up = nn.Upsample(scale_factor=2)
        if up_dim is None:
            #self.up = nn.ConvTranspose1d(n_inputs // 2, n_inputs // 2, kernel_size=4, stride=2, padding=1)#padding=1
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
        #print("**************************UPSAMPLING**************************")
        #print("original_x.shape:", x.shape)
        x = self.up(x)
        #print("upx1.shape:",x.shape)
        initial_x = x
        #print("t.shape:", t.shape)
        t = self.time_emb(t)  # .repeat(1, 1, x.shape[2])
        #print("t.shape:",t.shape)
        # print("x.shape:",x.shape)
        x = x + t
        if h is None:
            h = x
        #print("h.shape:",h.shape)
        #print("time_embedded_x.shape:", x.shape)
        shortcut = torch.cat([x, h], dim=1)
        #print("cat.shape:", shortcut.shape)
        # PREPARING SHORTCUT FEATURES
        shortcut = self.pre_shortcut_convs(shortcut)
        #print("pre_shortcut_convs.shape:", shortcut.shape)
        #shortcut = self.pre_shortcut_convs(x)
        shortcut = self.layer_norm1(shortcut)
        #print("layernorm1.shape:", shortcut.shape)
        shortcut = F.mish(shortcut)
        #print("F.mish.shape:", shortcut.shape)
        shortcut = self.shortcut_convs(shortcut)
        #print("shortcut_convs.shape:", shortcut.shape)
        shortcut = self.attention(shortcut)
        #print("attention.shape:", shortcut.shape)
        # shortcut = torch.cat([h, shortcut], dim=1)
        # PREPARING FOR DOWNSAMPLING
        out = self.post_shortcut_convs(shortcut)
        out = self.layer_norm2(out)
        #print("layernorm2.shape:", out.shape)
        out = F.mish(out)
        #print("F.mish.shape:", out.shape)
        a = self.res_conv(torch.cat([initial_x, h], dim=1))
        #print(a.shape)
        out = out + a
        #print("final_out.shape:", out.shape)
        return out





class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim,  number_of_diffusions,kernel_size = 5, n_heads = None, hidden_dim = None):
        cutted_channels = int((in_channels + out_channels) // 2) + 1
        super(DownsamplingBlock,self).__init__()
        self.kernel_size = kernel_size
        # self.time_emb = TimeEmbedding(2 * dim, number_of_diffusions, n_channels=in_channels // 2)
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
        # print("dim:",self.dim)
        # x = self.up(x)
        #print("---------------------------downsampling--------------------------------")
        initial_x = x
        #print("original_x.shape:",x.shape) # [batch_size x channel_num x signal_lenth]
        t = self.time_emb(t) #.repeat(1,1, x.shape[2])
        #print("t.shape:",t.shape)
        # print("x.shape:", x.shape)
        x = x + t
        #print("embedd_t_x.shape:", x.shape)

        #if h is None:
        #    h = x
        # shortcut = torch.cat([x, h], dim=2)
        # PREPARING SHORTCUT FEATURES
        # shortcut = self.conv1(shortcut)

        shortcut = self.conv1(x)
        #print("after_conv1_x.shape:",shortcut.shape)
        shortcut = self.layer_norm1(shortcut)
        #print("after_layernorm1_x.shape:", shortcut.shape)
        shortcut = F.mish(shortcut)
        #print("after_F.mish_x.shape:", shortcut.shape)
        shortcut = self.conv2(shortcut)
        #print("after_conv2_x.shape:", shortcut.shape)
        shortcut = self.attention(shortcut)
        #print("after_attention.shape:", shortcut.shape)
        # shortcut = torch.cat([h, shortcut], dim=1)
        # PREPARING FOR DOWNSAMPLING
        out = self.conv3(shortcut)
        out = self.layer_norm2(out)
        #print("after_layernorm2_x.shape:", out.shape)
        h = out
        out = self.down(out)
        #print("after_down_x.shape:", out.shape)
        out = F.mish(out)
        #print("final_out.shape:",out.shape)
        #temp = torch.cat([initial_x, h], dim=1)
        #print(temp.shape)
        # out = out + self.res_conv(torch.cat([initial_x, h], dim=1))
        # out = out + self.res_conv(torch.cat([initial_x, h], dim=2))

        return h, out



class ECGunetChannels(nn.Module):
    def __init__(self, number_of_diffusions, kernel_size=3, num_levels=3, n_channels=1, resolution=2048, device="cpu"):
        # def __init__(self, number_of_diffusions, resolution=512, kernel_size=3, num_levels=4, n_channels=1):
        super(ECGunetChannels, self).__init__()
        self.device = device

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

    def forward(self, x, t=None):  
        shortcuts = []
        
        # 确保输入数据在正确的设备上
        x = x.to(self.device)
        
        if t is None:
            t = torch.ones(x.shape[0], dtype=torch.long, device=self.device)
        else:
            t = t.to(self.device)
        
        
        out = x
        
        # DOWNSAMPLING BLOCKS
        for block in self.downsampling_blocks:
            h, out = block(out, t)
            shortcuts.append(h)
        
        del shortcuts[-1]
        
        old_out = out
        
        tt = self.time_emb(t)
        
        out = out + tt
        
        out = self.bottleneck_conv1(out)
        
        out = self.bottleneck_layer_norm1(out)
        
        out = F.mish(out)
        
        self_attention1 = self.attention_block(out)
        
        out = self.bottleneck_conv2(self_attention1)
        
        out = self.bottleneck_layer_norm2(out)
        
        out = F.mish(out) + old_out / math.sqrt(2)
        
        out = self.upsampling_blocks[0](out, None, t)
        
        for idx, block in enumerate(self.upsampling_blocks[1:]):
            out = block(out, shortcuts[-1 - idx], t)
        
        out = self.output_conv(out)
        
        return out
def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 模型参数
    number_of_diffusions = 1000
    kernel_size = 5
    num_levels = 3
    n_channels = 1
    resolution = 2048

    # 创建模型
    model = ECGunetChannels(
        number_of_diffusions=number_of_diffusions,
        kernel_size=kernel_size,
        num_levels=num_levels,
        n_channels=n_channels,
        resolution=resolution,
        device=device
    )
    

    # 加载预训练权重
    model.load_state_dict(torch.load("../weight/diffusion_unet.pth",map_location=device))

    # 冻结下采样部分的参数
    for param in model.downsampling_blocks.parameters():
        param.requires_grad = False

    # 调整上采样部分的输出层
    model.output_conv = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),  # 兼容不同长度信号
        nn.Flatten(),
        nn.Linear(1, 64),         # 输入维度自适应n_channels
        nn.Mish(),                # 保持与原始激活一致
        nn.Linear(64, 1)
    )
    model = model.to(device)

    # 创建随机输入数据
    batch_size = 2
    signal_length = resolution
    x = torch.randn(batch_size, n_channels, signal_length).to(device)

    # 前向传播
    output = model(x)

    # 打印输出形状
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()