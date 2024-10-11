import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super(MBConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.se(self.conv(x))
        else:
            return self.se(self.conv(x))

class ImprovedStableTransformationParamNet(nn.Module):
    def __init__(self, num_params=11):
        super(ImprovedStableTransformationParamNet, self).__init__()
        self.input_norm = nn.BatchNorm2d(3)

        self.block1 = MBConvBlock(3, 32, stride=2)
        self.block2 = MBConvBlock(32, 64, stride=2)
        self.block3 = MBConvBlock(64, 128, stride=2)
        self.block4 = MBConvBlock(128, 256, stride=2)

        # Add 1x1 convolutions for skip connections
        self.skip1 = nn.Conv2d(32, 256, 1)
        self.skip2 = nn.Conv2d(64, 256, 1)
        self.skip3 = nn.Conv2d(128, 256, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_params)
        self.ln = nn.LayerNorm(num_params)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        # Adjust skip connections
        x4 = x4 + F.interpolate(self.skip3(x3), size=x4.shape[2:], mode='bilinear', align_corners=False)
        x4 = x4 + F.interpolate(self.skip2(x2), size=x4.shape[2:], mode='bilinear', align_corners=False)
        x4 = x4 + F.interpolate(self.skip1(x1), size=x4.shape[2:], mode='bilinear', align_corners=False)

        x = self.pool(x4)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln(x)

        # Custom activation for each parameter
        params = torch.zeros_like(x)
        params[:, 0] = torch.tanh(x[:, 0])  # rotation: [-1, 1]
        params[:, 1:5] = torch.sigmoid(x[:, 1:5]) * 0.4  # crop params: [0, 0.4]
        params[:, 5:7] = torch.sigmoid(x[:, 5:7]) * 0.4 + 0.8  # brightness, contrast: [0.8, 1.2]
        params[:, 7:10] = torch.sigmoid(x[:, 7:10]) * 0.7 + 0.8  # color adjustments: [0.7, 1.5]
        params[:, 10] = torch.sigmoid(x[:, 10]) * 0.4 + 0.8  # resize: [0.8, 1.2]

        return params