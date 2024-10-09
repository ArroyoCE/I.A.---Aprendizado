import torch
import os
from PIL import Image
from torchvision import transforms
from neural_network import ImprovedStableTransformationParamNet
from apply_transformation import apply_transformations

def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

def process_images(input_dir, output_dir, checkpoint_path, device):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model and load the checkpoint
    model = ImprovedStableTransformationParamNet()
    model.to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    # Prepare image transformations
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Process each image in the input directory
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, f"transformed_{img_name}")

            # Load and preprocess the image
            input_image = Image.open(input_path).convert('RGB')
            input_tensor = preprocess(input_image).unsqueeze(0).to(device)

            # Get transformation parameters from the model
            with torch.no_grad():
                params = model(input_tensor)

            # Apply transformations
            original_tensor = transforms.ToTensor()(input_image).to(device)
            transformed_image = apply_transformations(original_tensor, params.squeeze())

            # Convert back to PIL Image and save
            transformed_image = transforms.ToPILImage()(transformed_image.cpu())
            transformed_image.save(output_path)

            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    src_dir = "../source_dir"
    dest_dir = "../output_images"
    ckpt_path = "./checkpoints/latest_checkpoint.pth"
    processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process_images(src_dir, dest_dir, ckpt_path, processing_device)