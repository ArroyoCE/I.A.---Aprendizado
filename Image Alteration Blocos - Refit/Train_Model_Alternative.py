import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchvision.io import read_image
from apply_transformation import apply_transformations
from pytorch_msssim import SSIM

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, target):
        h_x = x
        h_target = target
        h1_x = self.slice1(h_x)
        h1_target = self.slice1(h_target)
        h2_x = self.slice2(h1_x)
        h2_target = self.slice2(h1_target)
        h3_x = self.slice3(h2_x)
        h3_target = self.slice3(h2_target)
        h4_x = self.slice4(h3_x)
        h4_target = self.slice4(h3_target)
        loss = nn.functional.mse_loss(h1_x, h1_target) + nn.functional.mse_loss(h2_x, h2_target) + \
               nn.functional.mse_loss(h3_x, h3_target) + nn.functional.mse_loss(h4_x, h4_target)
        return loss


def train_model(model, train_loader, optimizer, device, checkpoint_dir, target_dir, num_epochs=100):
    model.to(device)
    mse_criterion = nn.MSELoss()
    ssim_criterion = SSIM(data_range=1.0, size_average=True, channel=3)
    perceptual_criterion = PerceptualLoss().to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    start_epoch = load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

        for batch_idx, (resized_tensor, original_tensor, img_name, original_size) in progress_bar:
            resized_tensor = resized_tensor.to(device)
            original_tensor = original_tensor.to(device)

            # Get corresponding target image
            img_name_str = str(img_name[0])
            target_image_path = os.path.join(target_dir, img_name_str)

            try:
                target_image = read_image(target_image_path).float() / 255.0
                target_image = target_image.unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error loading target image {img_name_str}: {str(e)}")
                continue

            optimizer.zero_grad()

            # Get parameters from the model using the resized tensor
            params = model(resized_tensor)

            if torch.isnan(params).any():
                print(f"\nNaN detected in params for image: {img_name_str}")
                continue

            # Apply transformations to the original tensor
            transformed_image = apply_transformations(original_tensor.squeeze(0), params.squeeze())
            transformed_image = transformed_image.unsqueeze(0)

            # Ensure both tensors have the same size
            if transformed_image.shape != target_image.shape:
                print(f"Size mismatch detected. Transformed: {transformed_image.shape}, Target: {target_image.shape}")
                min_height = min(transformed_image.shape[2], target_image.shape[2])
                min_width = min(transformed_image.shape[3], target_image.shape[3])

                transformed_image = F.interpolate(transformed_image, size=(min_height, min_width), mode='bilinear',
                                                  align_corners=False)
                target_image = F.interpolate(target_image, size=(min_height, min_width), mode='bilinear',
                                             align_corners=False)

                print(f"After resizing - Transformed: {transformed_image.shape}, Target: {target_image.shape}")

            # Sanity check
            assert transformed_image.shape == target_image.shape, f"Shapes still don't match after resizing: {transformed_image.shape} vs {target_image.shape}"

            # Calculate MSE and SSIM losses on matched size images
            mse_loss = mse_criterion(transformed_image, target_image)
            ssim_loss = 1 - ssim_criterion(transformed_image, target_image)  # 1 - SSIM because we want to minimize

            # Resize images for perceptual loss calculation
            max_size = 224  # You can adjust this value based on your VRAM capacity
            transformed_image_resized = F.interpolate(transformed_image, size=(max_size, max_size), mode='bilinear',
                                                      align_corners=False)
            target_image_resized = F.interpolate(target_image, size=(max_size, max_size), mode='bilinear',
                                                 align_corners=False)

            # Calculate perceptual loss on resized images
            perceptual_loss = perceptual_criterion(transformed_image_resized, target_image_resized)

            # Combine losses with weights
            total_loss = 0.2 * mse_loss + 0.3 * ssim_loss + 0.5 * perceptual_loss

            if torch.isnan(total_loss).item():
                print(f"\nNaN detected in loss for image: {img_name_str}")
                print(f"Original tensor range: [{original_tensor.min():.5f}, {original_tensor.max():.5f}]")
                print(f"Target image range: [{target_image.min():.5f}, {target_image.max():.5f}]")
                print(f"Transformed image range: [{transformed_image.min():.5f}, {transformed_image.max():.5f}]")
                print(f"Params: {params.detach().cpu().numpy()}")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()

            progress_bar.set_postfix({'Loss': f"{total_loss.item():.4f}"})

            # Save latest checkpoint every 10 items
            if (batch_idx + 1) % 10 == 0:
                latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / (batch_idx + 1),
                }, latest_checkpoint_path)

        epoch_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed, Loss: {epoch_loss:.4f}")

        # Step the scheduler
        scheduler.step(epoch_loss)

        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_end.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
            'loss': epoch_loss,
        }, checkpoint_path)

    print("\nTraining completed")

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