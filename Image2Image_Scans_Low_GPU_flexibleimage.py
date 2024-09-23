import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler


def to_float_tensor(x):
    return x.float()


class FlexibleDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FlexibleDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FlexibleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FlexibleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = FlexibleDoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), FlexibleDoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), FlexibleDoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), FlexibleDoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), FlexibleDoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = FlexibleDoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = FlexibleDoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = FlexibleDoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = FlexibleDoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

        # Camada de ajuste de tamanho com kernel e stride variáveis
        self.size_adjust = nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.crop_and_concat(x4, x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.crop_and_concat(x3, x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self.crop_and_concat(x2, x)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.crop_and_concat(x1, x)
        x = self.conv4(x)

        x = self.outc(x)
        x = self.size_adjust(x)
        return x

    def crop_and_concat(self, x1, x2):
        # Center crop x1 to match the size of x2
        diff_y = x1.size()[2] - x2.size()[2]
        diff_x = x1.size()[3] - x2.size()[3]
        x1 = x1[:, :, diff_y // 2:(diff_y // 2 + x2.size()[2]), diff_x // 2:(diff_x // 2 + x2.size()[3])]
        return torch.cat([x1, x2], dim=1)


class PairedImageDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        source_path = os.path.join(self.source_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)

        source_image = Image.open(source_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image


class FlexibleLoss(nn.Module):
    def __init__(self):
        super(FlexibleLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        # Encontra o tamanho mínimo entre output e target
        min_h = min(outputs.size(2), targets.size(2))
        min_w = min(outputs.size(3), targets.size(3))

        # Corta tanto o output quanto o target para o tamanho mínimo
        outputs = outputs[:, :, :min_h, :min_w]
        targets = targets[:, :, :min_h, :min_w]

        # Calcula a perda MSE
        loss = self.mse_loss(outputs, targets)

        # Retorna a média da perda
        return loss.mean()


def train_flexible_model(model, dataloader, criterion, optimizer, device, num_epochs, scaler, output_dir):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for source, target in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(source)
                loss = criterion(outputs, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(output_dir, "flexible_unet_model.pth"))
        print(f"Model saved after epoch {epoch + 1}")

    print("Training completed!")


# Uso principal
if __name__ == "__main__":
    # Configurações
    source_dir = "./source_dir"
    target_dir = "./target_dir"
    output_dir = "./output_dir"
    num_epochs = 100
    batch_size = 1
    learning_rate = 0.0001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_float_tensor)
    ])

    # Dataset e DataLoader
    dataset = PairedImageDataset(source_dir, target_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Configuração do modelo, perda e otimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexibleUNet(n_channels=3, n_classes=3).to(device)
    criterion = FlexibleLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = GradScaler()

    # Carrega modelo existente, se disponível
    if os.path.exists(os.path.join(output_dir, "flexible_unet_model.pth")):
        print("Loading existing model...")
        model.load_state_dict(torch.load(os.path.join(output_dir, "flexible_unet_model.pth")))
        print("Existing model loaded. Continuing training...")

    # Cria diretório de saída, se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_flexible_model(model, dataloader, criterion, optimizer, device, num_epochs, scaler, output_dir)
    print(f"Final model saved to {output_dir}")