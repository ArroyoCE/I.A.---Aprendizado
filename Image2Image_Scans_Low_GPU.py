import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler




def to_float_tensor(x):
    return x.float()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Calculate dynamic padding
        h, w = x.size()[-2:]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Encoder (with explicit checkpoint import)
        x1 = checkpoint(self.inc, x, use_reentrant=False, preserve_rng_state=False)
        x2 = checkpoint(self.down1, x1, use_reentrant=False, preserve_rng_state=False)
        x3 = checkpoint(self.down2, x2, use_reentrant=False, preserve_rng_state=False)
        x4 = checkpoint(self.down3, x3, use_reentrant=False, preserve_rng_state=False)
        x5 = checkpoint(self.down4, x4, use_reentrant=False, preserve_rng_state=False)

        # Decoder (without gradient checkpointing for simplicity)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)

        # Remove padding
        if pad_h > 0:
            logits = logits[:, :, :-pad_h, :]
        if pad_w > 0:
            logits = logits[:, :, :-pad_w]

        return logits

    def center_crop(self, source, target):
        source_shape = torch.tensor(source.shape)
        target_shape = torch.tensor(target.shape)
        diff = (source_shape[2:] - target_shape[2:]).div(2, rounding_mode='floor')
        return source[:, :, diff[0]:diff[0] + target_shape[2], diff[1]:diff[1] + target_shape[3]]


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


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        # Crop outputs to match targets size
        outputs = self.crop_tensor(outputs, targets.shape)
        mse = self.mse_loss(outputs, targets)
        l1 = self.l1_loss(outputs, targets)
        return self.alpha * mse + self.beta * l1

    def crop_tensor(self, tensor, target_shape):
        return tensor[..., :target_shape[2], :target_shape[3]]


def train_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device, num_epochs: int, scaler: GradScaler,
                output_dir: str, start_epoch: int = 0):
    for epoch in range(start_epoch, num_epochs):
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

        # Save the model after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(output_dir, "unet_model.pth"))
        print(f"Model saved after epoch {epoch + 1}")

    print("Training completed!")


# Main execution
if __name__ == "__main__":
    # Set up parameters
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

    # Create dataset and dataloader
    dataset = PairedImageDataset(source_dir, target_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Set up model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=3).to(device)
    criterion = CombinedLoss(alpha=0.5, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = GradScaler()

    # Check if there's an existing model to load
    if os.path.exists(os.path.join(output_dir, "unet_model.pth")):
        print("Loading existing model...")
        model.load_state_dict(torch.load(os.path.join(output_dir, "unet_model.pth")))
        print("Existing model loaded. Continuing training...")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_model(model, dataloader, criterion, optimizer, device, num_epochs, scaler, output_dir)
    print(f"Final model saved to {output_dir}")