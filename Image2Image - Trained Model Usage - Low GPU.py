import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

# Import the UNet model definition from the training script
from Image2Image_Scans_Low_GPU import UNet, to_float_tensor


class InferenceDataset(Dataset):
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


def apply_model(model, input_dir, output_dir, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_float_tensor)
    ])

    dataset = InferenceDataset(input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for batch, filenames in tqdm(dataloader, desc="Processing images"):
            batch = batch.to(device)

            outputs = model(batch)

            # Convert outputs to images and save
            for output, filename in zip(outputs, filenames):
                output = output.cpu().numpy().transpose(1, 2, 0)
                output = (output * 255).clip(0, 255).astype('uint8')
                output_image = Image.fromarray(output)
                output_path = os.path.join(output_dir, filename)
                output_image.save(output_path)


if __name__ == "__main__":
    # Set up parameters
    input_dir = "./input_images"
    output_dir = "./output_images"
    model_path = "./output_dir/unet_model.pth"
    batch_size = 1

    # Load the trained model
    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(torch.load(model_path))

    # Apply the model to the input images
    apply_model(model, input_dir, output_dir, batch_size)

    print(f"Processing complete. Output images saved to {output_dir}")