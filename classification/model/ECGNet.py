import torch
from torch import Tensor 
import torch.nn as nn
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint,self).__init__(*args,**kwargs)
    def forward(self, input: Tensor) -> Tensor:
        self.weight.data = torch.renorm(self.weight.data,p=2,dim=0,maxnorm=self.max_norm)
        return super(Conv2dWithConstraint,self).forward(input)
    
class ECGNet(nn.Module):
    def __init__(self, n_classes=2,channels=1,samples=2048,dropout=0.2,kernelLength=64, kernelLength2=16, F1=4,
                 D=2, F2=16,device='cpu') -> None:
        super(ECGNet,self).__init__()
        self.device = device
        self.F1=F1
        self.F2=F2
        self.D =D
        self.sample = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropout

        self.blocks = self.InitialBlocks(dropout).to(device)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, samples)
        self.classifierBlock = self.ClassifierBlock(self.F2 * self.blockOutputSize[1], n_classes).to(device)

    def InitialBlocks(self,dropout):
        block1 = nn.Sequential(
            nn.Conv2d(1,self.F1,(1,self.kernelLength), stride=1, padding=(0,self.kernelLength//2), bias=False),
            nn.BatchNorm2d(self.F1,momentum=0.01,affine=True,eps=1e-3),
            Conv2dWithConstraint(self.F1,self.F1*self.D , (self.channels, 1), max_norm=1,stride=1,padding=(0,0),groups=self.F1, bias=False),

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True),


            nn.ELU(),
            nn.AvgPool2d((1,4),stride=4),
            nn.Dropout(p=dropout),
        )
        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================


            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True),

            nn.Mish(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))
        block3 = nn.Sequential(
            nn.Conv2d(self.F2, 2*self.F2, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(2*self.F2, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True),
            nn.Mish(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))
        return nn.Sequential(block1, block2,block3)
    
    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Sequential(
            nn.Linear(inputSize, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes, bias=False),
        )

    
    def CalculateOutSize(self, model, channels, samples):
        device = next(model.parameters()).device
        data = torch.rand(1, 1, channels, samples).to(device)
        model.eval()
        with torch.no_grad():  # 避免计算梯度
            out = model(data).shape
        return out[2:]
    def forward(self, x):
        x = x.unsqueeze(2).permute(0,2,3,1)
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.classifierBlock(x)
        return x
