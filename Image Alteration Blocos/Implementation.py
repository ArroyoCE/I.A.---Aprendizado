import os
import torch
from torchvision import transforms
from PIL import Image
from neural_network import ImprovedStableTransformationParamNet
from apply_transformation import apply_transformations
from tqdm import tqdm


def load_model(checkpoint_path, device):
    model = ImprovedStableTransformationParamNet()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def process_image(model, image_path, device, resize_size=(512, 512)):
    # Load and preprocess the image
    input_image = Image.open(image_path).convert('RGB')
    resized_image = input_image.resize(resize_size, Image.Resampling.LANCZOS)
    dpi = input_image.info.get('dpi')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    resized_tensor = transform(resized_image).unsqueeze(0).to(device)
    original_tensor = transforms.ToTensor()(input_image).to(device)

    # Get parameters from the model
    with torch.no_grad():
        params = model(resized_tensor)

    # Apply transformations
    transformed_image = apply_transformations(original_tensor, params.squeeze())

    # Convert back to PIL Image
    transformed_pil = transforms.ToPILImage()(transformed_image.cpu().clamp(0, 1))

    if dpi:
        transformed_pil.info['dpi'] = dpi

    return transformed_pil




def main():
    # Set up paths and device
    input_dir = "../Pix2Pix/pytorch-CycleGAN-and-pix2pix/datasets/scans/train"
    output_dir = "../output_images"
    checkpoint_path = "./checkpoints/latest_checkpoint.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the trained model
    model = load_model(checkpoint_path, device)

    # Process all images in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"transformed_{image_file}")

        try:
            transformed_image = process_image(model, input_path, device)

            # Save the image with original PPI (DPI) information
            dpi = transformed_image.info.get('dpi')
            if dpi:
                transformed_image.save(output_path, dpi=dpi)
            else:
                transformed_image.save(output_path)
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    print(f"Processed {len(image_files)} images. Results saved in {output_dir}")


if __name__ == "__main__":
    main()