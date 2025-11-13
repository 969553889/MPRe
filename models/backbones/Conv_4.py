import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,input_channel,output_channel):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel))
    def forward(self,inp):
        return self.layers(inp)

class BackBone(nn.Module):
    def __init__(self,num_channel=64):
        super().__init__()
        self.layer1 = nn.Sequential(ConvBlock(3, num_channel), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(ConvBlock(num_channel, num_channel), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(ConvBlock(num_channel, num_channel), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(ConvBlock(num_channel, num_channel), nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self,inp, return_intermediate=False):
        l1 = self.layer1(inp)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)  # 作为“中层”
        l4 = self.layer4(l3)  # 最后卷积层
        return l4, l3
