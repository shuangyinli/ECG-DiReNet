import torch
import torch.nn as nn
from model.DiffusionUnet import ECGunetChannels
from model.ECGNet import ECGNet


class Model(nn.Module):
    def __init__(self, device, number_of_diffusions=1000, kernel_size=5, num_levels=3, n_channels=1, resolution=2048,load_path=None):
        super().__init__()
        self.device = device
        self.unet = ECGunetChannels(
            number_of_diffusions=number_of_diffusions,
            kernel_size=kernel_size,
            num_levels=num_levels,
            n_channels=n_channels,
            resolution=resolution,
            device=device
        )
 
        self.ecgnet = ECGNet(n_classes=1, channels=1, samples=2048, dropout=0,
                                  kernelLength=64, kernelLength2=16, F1=8, D=2, F2=16, device=device).to(device)
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        if load_path != None:
            self.load_weights(load_path)
        self.freeze_downsampling()
        self.adjust_output_conv()

    def load_weights(self,load_path):
        self.unet.load_state_dict(torch.load(load_path, map_location=self.device))

    def freeze_downsampling(self):
        for param in self.unet.downsampling_blocks.parameters():
            param.requires_grad = False

    def adjust_output_conv(self):
        self.unet.output_conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   
            nn.Flatten(),
            nn.Linear(1, 64),  
            nn.Mish(),  
            nn.Linear(64, 1)
        )
        self.unet = self.unet.to(self.device)

    def forward(self, x):
        unet_input = x.permute(0, 2, 1)
        unet_out = self.unet(unet_input)

        ecg_out = self.ecgnet(x)

        # Weighted fusion
        weighted = self.weights[0] * unet_out + self.weights[1] * ecg_out
        return weighted

def main():
    # Usage example
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(device)
    # Generate random input data for testing
    input_data = torch.randn(16, 2048, 1).to(device)
    output = model(input_data)
    print(output.shape)

if __name__ == "__main__":
    main()