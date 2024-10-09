import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SourceImageDataset(Dataset):
    def __init__(self, input_dir, resize_size=(512, 512)):
        self.input_dir = input_dir
        self.image_files = [image for image in os.listdir(input_dir) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.resize_size = resize_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.reshuffle()  # Shuffle the image files when initializing the dataset

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_image = Image.open(os.path.join(self.input_dir, img_name)).convert('RGB')

        # Store original size
        original_size = input_image.size

        # Create resized version for network input
        resized_image = input_image.resize(self.resize_size, Image.Resampling.LANCZOS)

        # Convert both to tensors
        original_tensor = transforms.ToTensor()(input_image)
        resized_tensor = transforms.ToTensor()(resized_image)

        # Normalize only the resized tensor (for network input)
        resized_tensor = self.transform(resized_tensor)

        return resized_tensor, original_tensor, img_name, original_size

    def reshuffle(self):
        random.shuffle(self.image_files)