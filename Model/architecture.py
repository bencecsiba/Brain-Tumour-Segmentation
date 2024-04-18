import torch.nn as nn
import torch

def convblock(in_channels, out_channels):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)

    )

def downconv(in_channels, out_channels):
    
    return nn.Sequential(

        convblock(in_channels, out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2)

    )
   
def upconv(in_channels, out_channels):

    return nn.Sequential(

        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
        convblock(in_channels, out_channels),

    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        #input_dim = 256
        self.encoder = nn.Sequential(
            downconv(in_channels, 64), #128
            downconv(64, 128), #64
            downconv(128, 256), #32
            downconv(256, 512) #16
        )
        
        self.bottleneck = convblock(512, 1024)
        
        #extra channels allow for concatenation of skip connections in upsampling block
        self.decoder = nn.ModuleList([
            upconv(512+1024,512), #32
            upconv(256+512,256), #64
            upconv(128+256,128), #128
            upconv(64+128,64) #256
        ])
        
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        skips = []
        
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        
        x = self.bottleneck(x)
        
        for i, layer in enumerate(self.decoder):

            x = torch.cat((skips[len(skips)-i-1],x), dim=1)

            x = layer(x)
        
        return self.output_conv(x)