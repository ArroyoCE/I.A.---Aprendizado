import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import os
from tqdm import tqdm
from torch.autograd import Function
import random
import math

class GradientClippingFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, *grad_outputs):
        return tuple(torch.clamp(g, -1, 1) if g is not None else None for g in grad_outputs)


clip_gradients = GradientClippingFunction.apply


class StableTransformationParamNet(nn.Module):
    def __init__(self):
        super(StableTransformationParamNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 7)
        self.ln = nn.LayerNorm(7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)

        params = torch.zeros_like(x)
        params[:, 0] = torch.tanh(x[:, 0]) * 45  # Rotation: [-45, 45]
        params[:, 1:5] = torch.sigmoid(x[:, 1:5]) * 0.4  # Crop: [0, 0.4]
        params[:, 5:7] = torch.sigmoid(x[:, 5:7]) * 0.2 + 0.9  # Color levels: [0.9, 1.1]

        return params


def stable_mse_loss(input, target):
    return torch.mean(torch.square(input - target))


def custom_collate(batch):
    input_images, target_images, img_names = zip(*batch)
    return list(input_images), list(target_images), list(img_names)


class ImageAlterationDataset(Dataset):
    def __init__(self, input_dir, target_dir, resize_size=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_files = [image for image in os.listdir(input_dir) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.resize_size = resize_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_image = Image.open(os.path.join(self.input_dir, img_name)).convert('RGB')
        target_image = Image.open(os.path.join(self.target_dir, img_name)).convert('RGB')

        if self.resize_size:
            input_image = input_image.resize(self.resize_size, Image.Resampling.LANCZOS)
            target_image = target_image.resize(self.resize_size, Image.Resampling.LANCZOS)
        else:
            # Resize images to match the smaller one if no specific size is given
            input_size = input_image.size
            target_size = target_image.size
            min_size = (min(input_size[0], target_size[0]), min(input_size[1], target_size[1]))
            input_image = input_image.resize(min_size, Image.Resampling.LANCZOS)
            target_image = target_image.resize(min_size, Image.Resampling.LANCZOS)

        # Convert to tensors
        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

        return input_tensor, target_tensor, img_name

    def reshuffle(self):
        random.shuffle(self.image_files)


def apply_transformations(image, params):
    device = params.device
    original_size = image.shape[-2:]  # Store original size
    image = image.unsqueeze(0).to(device)

    rotation, crop_left, crop_right, crop_top, crop_bottom, brightness, contrast = params

    # Apply rotation
    angle = rotation * 180 / math.pi
    image = TF.rotate(image, angle.item(), interpolation=TF.InterpolationMode.BILINEAR)

    # Apply crop
    _, _, h, w = image.shape
    left = int(crop_left.item() * w)
    top = int(crop_top.item() * h)
    right = w - int(crop_right.item() * w)
    bottom = h - int(crop_bottom.item() * h)
    image = TF.crop(image, top, left, bottom - top, right - left)

    # Resize back to original size
    image = TF.resize(image, original_size, interpolation=TF.InterpolationMode.BILINEAR)

    # Apply color adjustments
    image = image * brightness
    image = (image - 0.5) * contrast + 0.5

    return image.squeeze(0)


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'. Starting from scratch.")
        return 0


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved to '{filename}'")


def train_model(model, train_loader, optimizer, scheduler, num_epochs, device, checkpoint_dir):
    model.to(device)
    criterion = stable_mse_loss

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_loader.dataset.reshuffle()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

        for batch_idx, (input_tensors, target_tensors, img_names) in progress_bar:
            batch_loss = 0.0

            for input_tensor, target_tensor, img_name in zip(input_tensors, target_tensors, img_names):
                input_tensor = input_tensor.unsqueeze(0).to(device).requires_grad_(True)
                target_tensor = target_tensor.unsqueeze(0).to(device)

                optimizer.zero_grad()

                params = model(input_tensor)

                if torch.isnan(params).any():
                    print(f"\nNaN detected in params for image: {img_name}")
                    continue

                transformed_tensor = apply_transformations(input_tensor.squeeze(0), params.squeeze())

                # Ensure both tensors have the same size
                if transformed_tensor.shape != target_tensor.shape:
                    min_height = min(transformed_tensor.shape[-2], target_tensor.shape[-2])
                    min_width = min(transformed_tensor.shape[-1], target_tensor.shape[-1])
                    transformed_tensor = TF.center_crop(transformed_tensor, [min_height, min_width])
                    target_tensor = TF.center_crop(target_tensor, [min_height, min_width])

                loss = criterion(transformed_tensor, target_tensor)

                if torch.isnan(loss).item():
                    print(f"\nNaN detected in loss for image: {img_name}")
                    print(f"Input tensor range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
                    print(f"Target tensor range: [{target_tensor.min():.4f}, {target_tensor.max():.4f}]")
                    print(f"Transformed tensor range: [{transformed_tensor.min():.4f}, {transformed_tensor.max():.4f}]")
                    print(f"Params: {params.detach().cpu().numpy()}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss += loss.item()

            avg_batch_loss = batch_loss / len(input_tensors)
            running_loss += avg_batch_loss

            progress_bar.set_postfix({'Loss': f"{avg_batch_loss:.4f}"})

        epoch_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed, Loss: {epoch_loss:.4f}")

        scheduler.step(epoch_loss)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)

    print("\nTraining completed")


def main():
    input_dir = "./source_dir"
    target_dir = "./target_dir"
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = StableTransformationParamNet()
    dataset = ImageAlterationDataset(input_dir, target_dir, resize_size=(512, 512))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))

    train_model(model, dataloader, optimizer, scheduler, num_epochs=100 - start_epoch, device=device,
                checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()