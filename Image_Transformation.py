import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
from PIL import Image
import os


class TransformationParamNet(nn.Module):
    def __init__(self):
        super(TransformationParamNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 9)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ImageAlterationDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=(5100, 8400)):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_files = [F for F in os.listdir(input_dir) if F.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_image = Image.open(os.path.join(self.input_dir, img_name)).convert('RGB')
        target_image = Image.open(os.path.join(self.target_dir, img_name)).convert('RGB')

        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

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
    grid = f.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
    return f.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)


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

    grid = f.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
    return f.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)


def adjust_brightness(image, brightness):
    return torch.clamp(image * brightness, 0, 1)


def adjust_contrast(image, contrast):
    mean = image.mean(dim=[-3, -2, -1], keepdim=True)
    return torch.clamp((image - mean) * contrast + mean, 0, 1)


def adjust_saturation(image, saturation):
    gray = image.mean(dim=0, keepdim=True)
    return torch.clamp((image - gray) * saturation + gray, 0, 1)


def rgb_to_hsv(rgb):
    r, g, b = rgb.unbind(0)
    max_rgb, argmax_rgb = rgb.max(0)
    min_rgb, _ = rgb.min(0)
    diff = max_rgb - min_rgb

    s = torch.where(max_rgb > 0, diff / max_rgb, torch.zeros_like(max_rgb))
    v = max_rgb

    h = torch.zeros_like(max_rgb)
    h[argmax_rgb == 0] = (g - b)[argmax_rgb == 0] / diff[argmax_rgb == 0]
    h[argmax_rgb == 1] = 2.0 + (b - r)[argmax_rgb == 1] / diff[argmax_rgb == 1]
    h[argmax_rgb == 2] = 4.0 + (r - g)[argmax_rgb == 2] / diff[argmax_rgb == 2]
    h[diff == 0] = 0
    h = (h / 6.0) % 1.0

    return torch.stack([h, s, v])


def hsv_to_rgb(hsv):
    h, s, v = hsv.unbind(0)
    i = (h * 6).long()
    l = h * 6 - i.float()
    p = v * (1 - s)
    q = v * (1 - l * s)
    t = v * (1 - (1 - l) * s)
    i = i % 6

    rgb = torch.stack([
        torch.where(i == 0, v,
                    torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v))))),
        torch.where(i == 0, t,
                    torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p))))),
        torch.where(i == 0, p,
                    torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    ])

    return rgb


def adjust_hue(image, hue):
    # Ensure hue is in the range [-0.5, 0.5]
    hue = torch.clamp(hue, -0.5, 0.5)

    # Convert to HSV
    hsv = rgb_to_hsv(image)

    # Adjust hue
    h, s, v = hsv.unbind(0)
    h = (h + hue) % 1.0
    hsv = torch.stack([h, s, v])

    # Convert back to RGB
    rgb = hsv_to_rgb(hsv)

    return rgb


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            params = model.forward(inputs)
            altered_image = apply_transformations(inputs.squeeze(0), params.squeeze(0), device)
            loss = criterion(altered_image.unsqueeze(0), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


def main():
    input_dir = "./source_dir"
    target_dir = "./target_dir"
    image_size = (5100, 8400)  # Maintaining the original image size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    model = TransformationParamNet().to(device)
    dataset = ImageAlterationDataset(input_dir, target_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Keeping batch size at 1

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=50, device=device)

    torch.save(model.state_dict(), "image_alteration_model.pth")


if __name__ == "__main__":
    main()