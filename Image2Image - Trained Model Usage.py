import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# U-Net model definition (same as in the training script)
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
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x + x4)
        x = self.up3(x + x3)
        x = self.up4(x + x2)
        logits = self.outc(x + x1)
        return logits

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def translate_image(model, image_tensor, device):
    with torch.no_grad():
        output = model(image_tensor.to(device))
    return output.squeeze(0).cpu()  # Remove batch dimension

def tensor_to_image(tensor):
    tensor = tensor.clamp(-1, 1)  # Ensure the tensor is in the range [-1, 1]
    tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
    tensor = tensor.permute(1, 2, 0)  # Change from [C, H, W] to [H, W, C]
    return (tensor.numpy() * 255).astype('uint8')

def process_folder(model, input_folder, output_folder, transform, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"translated_{image_file}")

        image_tensor = process_image(input_path, transform)
        translated_tensor = translate_image(model, image_tensor, device)
        translated_image = tensor_to_image(translated_tensor)

        Image.fromarray(translated_image).save(output_path)

if __name__ == "__main__":
    # Set up parameters
    model_path = "./output_dir/unet_model.pth"
    input_folder = "./input_images"
    output_folder = "./output_images"

    # Set up transform (same as in training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model(model_path, device)

    # Process the folder
    process_folder(model, input_folder, output_folder, transform, device)

    print(f"Processing complete. Translated images saved in {output_folder}")