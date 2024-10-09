import torch
import torch.nn as nn
import torch.nn.functional as f

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = f.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = f.leaky_relu(out, negative_slope=0.2)
        return out

class ImprovedStableTransformationParamNet(nn.Module):
    def __init__(self, num_params=10):
        super(ImprovedStableTransformationParamNet, self).__init__()
        self.input_norm = nn.BatchNorm2d(3)
        self.conv1 = ResidualBlock(3, 32)
        self.conv2 = ResidualBlock(32, 64)
        self.conv3 = ResidualBlock(64, 128)
        self.conv4 = ResidualBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_params)
        self.ln = nn.LayerNorm(num_params)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.conv1(x)
        x = f.max_pool2d(x, 2)
        x = self.conv2(x)
        x = f.max_pool2d(x, 2)
        x = self.conv3(x)
        x = f.max_pool2d(x, 2)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = f.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln(x)

        # Custom activation for each parameter
        params = torch.zeros_like(x)
        params[:, 0] = torch.tanh(x[:, 0]) * 5  # rotation: [-5, 5]
        params[:, 1:5] = torch.sigmoid(x[:, 1:5]) * 0.4  # crop params: [0, 0.4]
        params[:, 5:7] = torch.sigmoid(x[:, 5:7]) * 0.4 + 0.8  # brightness, contrast: [0.8, 1.2]
        params[:, 7:10] = torch.sigmoid(x[:, 7:10]) * 0.2 + 0.9  # color adjustments: [0.9, 1.1]

        return params