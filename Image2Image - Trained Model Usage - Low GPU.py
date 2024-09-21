import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import the UNet model definition
from unet_model import UNet


class SingleImageDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.input_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def modify_images(model, dataloader, device, output_dir):
    model.eval()

    with torch.no_grad():
        for inputs, img_names in tqdm(dataloader, desc="Processing images"):
            inputs = inputs.to(device)

            outputs = model(inputs)

            # Convert outputs to image format
            outputs = outputs.cpu().squeeze(0).permute(1, 2, 0).numpy()
            outputs = (outputs * 255).clip(0, 255).astype('uint8')

            # Save the output image
            for output, img_name in zip(outputs, img_names):
                output_image = Image.fromarray(output)
                output_path = os.path.join(output_dir, f"modified_{img_name}")
                output_image.save(output_path)


if __name__ == "__main__":
    # Set up parameters
    input_dir = "./input_dir"  # Directory containing images to be modified
    output_dir = "./output_dir"  # Directory to save modified images
    model_path = "./unet_model.pth"  # Path to the trained model
    batch_size = 1

    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = SingleImageDataset(input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=3).to(device)

    # Load the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Modify images
    modify_images(model, dataloader, device, output_dir)

    print(f"Modified images saved to {output_dir}")