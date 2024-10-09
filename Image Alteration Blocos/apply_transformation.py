import torch
import torchvision.transforms.functional as TF

def apply_transformations(
    image: torch.Tensor,
    params: torch.Tensor
) -> torch.Tensor:
    """
    Apply various transformations to an image based on neural network output.

    Args:
        image (torch.Tensor): The input image tensor of shape (C, H, W).
        params (torch.Tensor): The transformation parameters tensor of shape (11,).

    Returns:
        torch.Tensor: The transformed image tensor.
    """
    rotation, crop_left, crop_top, crop_right, crop_bottom, brightness, contrast, red_adjust, green_adjust, blue_adjust, resize_param = params

    # Ensure the image is a float tensor
    image = image.float()

    # Get original dimensions
    _, original_height, original_width = image.shape

    # Apply rotation
    if rotation != 0:
        angle = rotation.item() / torch.pi  # Convert to degrees, but reduce the range
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, expand=True)

    # Apply crop
    if torch.any(torch.tensor([crop_left, crop_top, crop_right, crop_bottom]) != 0):
        _, height, width = image.shape
        left = int(crop_left * width * 0.25)  # Reduce crop effect
        top = int(crop_top * height * 0.25)
        right = width - int(crop_right * width * 0.25)
        bottom = height - int(crop_bottom * height * 0.25)
        image = TF.crop(image, top, left, bottom - top, right - left)

    # Apply brightness and contrast adjustments
    if brightness != 1.0 or contrast != 1.0:
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)

    # Apply color level adjustments
    if torch.any(torch.tensor([red_adjust, green_adjust, blue_adjust]) != 1.0):
        r, g, b = image.unbind(0)
        r = torch.clamp(r * red_adjust, 0, 1)
        g = torch.clamp(g * green_adjust, 0, 1)
        b = torch.clamp(b * blue_adjust, 0, 1)
        image = torch.stack([r, g, b])

    # Apply resize (translate resize_param to actual resize factor)
    resize_factor = 1.0 - (resize_param - 0.8) / 0.4 * 0.3  # Map 0.8-1.2 to 1.0-0.6
    if resize_factor < 1.0:
        _, current_height, current_width = image.shape
        new_height = int(current_height * resize_factor)
        new_width = int(current_width * resize_factor)
        image = TF.resize(image, [new_height, new_width], interpolation=TF.InterpolationMode.BILINEAR)

    return image