import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if "Conv" in classname: 
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm2d" in classname or "InstanceNorm2d" in classname:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.key = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.value = spectral_norm(nn.Conv2d(in_dim, in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, width, height = x.size()
        proj_query = self.query(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, width, height)
        out = self.gamma * out + x
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_features, in_features, kernel_size=3)),
            nn.InstanceNorm2d(in_features, affine=True), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_features, in_features, kernel_size=3)),
            nn.InstanceNorm2d(in_features, affine=True),
        )
    
    def forward(self, x):
        return x + self.block(x)
    

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(channels, out_features, kernel_size=7)),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                spectral_norm(nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1)),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        
        model += [
            spectral_norm(nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        model += [SelfAttention(out_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [
            spectral_norm(nn.Conv2d(in_features, out_features, kernel_size= 3, stride=1, padding=1)),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.ReLU(inplace=True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(out_features, channels, kernel_size=7)),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True)),
            layers.append(nn.Dropout(0.3))
            return layers

        model = [
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        ]

        model += [
            spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [SelfAttention(512)]

        model += [
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, padding=1)),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, img):
        return self.model(img)
