import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TransformationParamNet(nn.Module):
    def __init__(self):
        super(TransformationParamNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 13)
        self.dropout = nn.Dropout(0.3)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        rotation = torch.tanh(x[:, 0]).unsqueeze(1) * 180
        crop_params = torch.sigmoid(x[:, 1:5])
        brightness = torch.sigmoid(x[:, 5]).unsqueeze(1) * 2
        contrast = torch.sigmoid(x[:, 6]).unsqueeze(1) * 2
        saturation = torch.sigmoid(x[:, 7]).unsqueeze(1) * 2
        hue = torch.tanh(x[:, 8]).unsqueeze(1) * 0.5
        shear_x = torch.tanh(x[:, 9]).unsqueeze(1) * 0.5
        shear_y = torch.tanh(x[:, 10]).unsqueeze(1) * 0.5
        translate_x = torch.tanh(x[:, 11]).unsqueeze(1) * 0.2
        translate_y = torch.tanh(x[:, 12]).unsqueeze(1) * 0.2

        return torch.cat(
            [rotation, crop_params, brightness, contrast, saturation, hue, shear_x, shear_y, translate_x, translate_y],
            dim=1)

class ImageAlterationDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_files = [image for image in os.listdir(input_dir) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_image = Image.open(os.path.join(self.input_dir, img_name)).convert('RGB')
        target_image = Image.open(os.path.join(self.target_dir, img_name)).convert('RGB')

        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

        return input_tensor, target_tensor, input_image.size


def apply_transformations(image, params):
    rotation, crop_x, crop_y, crop_w, crop_h, brightness, contrast, saturation, hue, shear_x, shear_y, translate_x, translate_y = params


    # Apply affine transformations (rotation, shear, translation)
    image = affine_transform(image, rotation, shear_x, shear_y, translate_x, translate_y)

    # Crop
    image = differentiable_crop(image, crop_x, crop_y, crop_w, crop_h)

    # Color adjustments
    image = adjust_brightness(image, brightness)
    image = adjust_contrast(image, contrast)
    image = adjust_saturation(image, saturation)
    image = adjust_hue(image, hue)

    return torch.clamp(image, 0, 1)


def affine_transform(image, angle, shear_x, shear_y, translate_x, translate_y):
    device = image.device

    # Convert angle to radians
    angle = angle * torch.pi / 180

    # Create affine transformation matrix
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    affine_matrix = torch.tensor([
        [cos_theta, -sin_theta + shear_x, translate_x],
        [sin_theta + shear_y, cos_theta, translate_y]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    # Apply affine transformation
    grid = F.affine_grid(affine_matrix, image.unsqueeze(0).size(), align_corners=False)
    transformed = F.grid_sample(image.unsqueeze(0), grid, align_corners=False, padding_mode='border')

    return transformed.squeeze(0)


def differentiable_crop(image, crop_x, crop_y, crop_w, crop_h):
    device = image.device
    _, h, w = image.shape

    x1 = int(crop_x * w)
    y1 = int(crop_y * h)
    x2 = int((crop_x + crop_w) * w)
    y2 = int((crop_y + crop_h) * h)

    # Ensure valid crop dimensions
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    # Create the cropping grid
    theta = torch.tensor([
        [2.0 / (x2 - x1), 0, -1.0 - 2.0 * x1 / (x2 - x1)],
        [0, 2.0 / (y2 - y1), -1.0 - 2.0 * y1 / (y2 - y1)]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    grid = F.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
    cropped = F.grid_sample(image.unsqueeze(0), grid, align_corners=False, padding_mode='border')

    return cropped.squeeze(0)


def adjust_brightness(image, brightness):
    return image * brightness


def adjust_contrast(image, contrast):
    mean = image.mean(dim=[1, 2], keepdim=True)
    return (image - mean) * contrast + mean


def adjust_saturation(image, saturation):
    gray = image.mean(dim=0, keepdim=True)
    return (image - gray) * saturation + gray


def adjust_hue(image, hue):
    r, g, b = image.unbind(0)

    # Convert RGB to HSV
    max_rgb, argmax_rgb = image.max(0)
    min_rgb, _ = image.min(0)
    diff = max_rgb - min_rgb

    s = torch.where(max_rgb > 0, diff / (max_rgb + 1e-8), torch.zeros_like(max_rgb))
    v = max_rgb

    # Calculate hue
    eps = 1e-8
    h_tmp = torch.stack([
        (g - b) / (diff + eps),
        2.0 + (b - r) / (diff + eps),
        4.0 + (r - g) / (diff + eps)
    ])

    h = h_tmp.gather(0, argmax_rgb.unsqueeze(0)).squeeze(0)

    # Create a mask for diff == 0
    mask = (diff > eps).float()
    h = h * mask

    h = (h / 6.0) % 1.0

    # Adjust hue
    h = (h + hue) % 1.0

    # Convert back to RGB
    hi = (h * 6).long()
    f = (h * 6) - hi.float()
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    hi = hi % 6

    rgb = torch.stack([
        torch.where(hi == 0, v, torch.where(hi == 1, q, torch.where(hi == 2, p, torch.where(hi == 3, p,
                                                                                            torch.where(hi == 4, t,
                                                                                                        v))))),
        torch.where(hi == 0, t, torch.where(hi == 1, v, torch.where(hi == 2, v, torch.where(hi == 3, q,
                                                                                            torch.where(hi == 4, p,
                                                                                                        p))))),
        torch.where(hi == 0, p, torch.where(hi == 1, p, torch.where(hi == 2, t, torch.where(hi == 3, v,
                                                                                            torch.where(hi == 4, v,
                                                                                                        q)))))
    ])

    return rgb





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


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, checkpoint_dir):
    model.to(device)

    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (inputs, targets, sizes) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            params = model(inputs)

            altered_images = []
            for img, param, size in zip(inputs, params, sizes):
                altered_image = apply_transformations(img, param)
                altered_image = F.interpolate(altered_image.unsqueeze(0), size=size, mode='bilinear',
                                              align_corners=False).squeeze(0)
                altered_images.append(altered_image)

            altered_images = torch.stack(altered_images)
            loss = criterion(altered_images, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")

        # Step the scheduler
        scheduler.step(epoch_loss)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)

    print("Training completed")

def main():
    input_dir = "./source_dir"
    target_dir = "./target_dir"
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformationParamNet().to(device)
    dataset = ImageAlterationDataset(input_dir, target_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=50, device=device, checkpoint_dir=checkpoint_dir)

if __name__ == "__main__":
    main()