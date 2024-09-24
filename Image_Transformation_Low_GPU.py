import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import os
from torch.cuda.amp import autocast, GradScaler


class TransformationParamNet(nn.Module):
    def __init__(self):
        super(TransformationParamNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 9)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PatchedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=(5100, 8400), patch_size=(510, 840)):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = image_size
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files) * (
                    (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1]))

    def __getitem__(self, idx):
        img_idx = idx // ((self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1]))
        patch_idx = idx % ((self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1]))

        img_name = self.image_files[img_idx]
        input_image = Image.open(os.path.join(self.input_dir, img_name)).convert('RGB')
        target_image = Image.open(os.path.join(self.target_dir, img_name)).convert('RGB')

        row = (patch_idx // (self.image_size[1] // self.patch_size[1])) * self.patch_size[0]
        col = (patch_idx % (self.image_size[1] // self.patch_size[1])) * self.patch_size[1]

        input_patch = input_image.crop((col, row, col + self.patch_size[1], row + self.patch_size[0]))
        target_patch = target_image.crop((col, row, col + self.patch_size[1], row + self.patch_size[0]))

        input_tensor = self.transform(input_patch)
        target_tensor = self.transform(target_patch)

        return input_tensor, target_tensor


def apply_transformations(image, params, device):
    rotation, crop_x, crop_y, crop_w, crop_h, brightness, contrast, saturation, hue = params

    angle = rotation * 360
    image = rotate_image(image, angle, device)
    image = differentiable_crop(image, crop_x, crop_y, crop_w, crop_h, device)
    image = adjust_brightness(image, brightness + 1)
    image = adjust_contrast(image, contrast + 1)
    image = adjust_saturation(image, saturation + 1)
    image = adjust_hue(image, hue)

    return image


def rotate_image(image, angle, device):
    angle = angle * torch.pi / 180
    theta = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0]
    ], dtype=torch.float, device=device)
    grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
    return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)


def differentiable_crop(image, crop_x, crop_y, crop_w, crop_h, device):
    _, h, w = image.shape
    crop_x = crop_x * w
    crop_y = crop_y * h
    crop_w = crop_w * w
    crop_h = crop_h * h

    x1 = torch.clamp(crop_x, 0, w - 1)
    y1 = torch.clamp(crop_y, 0, h - 1)
    x2 = torch.clamp(x1 + crop_w, 0, w)
    y2 = torch.clamp(y1 + crop_h, 0, h)

    theta = torch.tensor([
        [2 / (x2 - x1), 0, -1 - 2 * x1 / (x2 - x1)],
        [0, 2 / (y2 - y1), -1 - 2 * y1 / (y2 - y1)]
    ], dtype=torch.float, device=device)

    grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
    return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)


def adjust_brightness(image, brightness):
    return torch.clamp(image * brightness, 0, 1)


def adjust_contrast(image, contrast):
    mean = image.mean(dim=[-3, -2, -1], keepdim=True)
    return torch.clamp((image - mean) * contrast + mean, 0, 1)


def adjust_saturation(image, saturation):
    gray = image.mean(dim=0, keepdim=True)
    return torch.clamp((image - gray) * saturation + gray, 0, 1)


def adjust_hue(image, hue):
    rgb_to_hsv = torch.tensor([
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114]
    ], device=image.device)

    hsv_to_rgb = torch.tensor([
        [1, 0, 1],
        [-0.14713, 0.28886, -0.14713],
        [0.615, -0.51499, -0.10001]
    ], device=image.device)

    hsv = torch.matmul(rgb_to_hsv, image.permute(1, 2, 0))
    hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 1.0
    rgb = torch.matmul(hsv_to_rgb, hsv.permute(2, 0, 1))
    return torch.clamp(rgb.permute(1, 2, 0), 0, 1)

@torch.no_grad()
def apply_model(model, image, patch_size, device):
    _, h, w = image.shape
    params = torch.zeros(9, device=device)
    count = 0
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[:, i:i + patch_size[0], j:j + patch_size[1]].unsqueeze(0)
            params += model(patch).squeeze(0)
            count += 1
    return params / count


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, accumulation_steps=4):
    model.to(device)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                params = model(inputs)
                altered_image = apply_transformations(inputs.squeeze(0), params.squeeze(0), device)
                loss = criterion(altered_image.unsqueeze(0), targets)

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


def main():
    input_dir = "./source_dir"
    target_dir = "./target_dir"
    image_size = (5100, 8400)
    patch_size = (510, 840)  # Process 1/10 of the image at a time

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformationParamNet().to(device)
    dataset = PatchedImageDataset(input_dir, target_dir, image_size=image_size, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=50, device=device, accumulation_steps=4)

    torch.save(model.state_dict(), "image_alteration_model.pth")


if __name__ == "__main__":
    main()