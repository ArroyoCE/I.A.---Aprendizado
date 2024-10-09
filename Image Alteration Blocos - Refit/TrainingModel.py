import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from apply_transformation import apply_transformations


def train_model(model, train_loader, optimizer, device, checkpoint_dir, target_dir, num_epochs=100):
    model.to(device)
    criterion = nn.MSELoss()
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
                min_height = min(transformed_image.shape[-2], target_image.shape[-2])
                min_width = min(transformed_image.shape[-1], target_image.shape[-1])
                transformed_image = TF.center_crop(transformed_image, [min_height, min_width])
                target_image = TF.center_crop(target_image, [min_height, min_width])

            loss = criterion(transformed_image, target_image)

            if torch.isnan(loss).item():
                print(f"\nNaN detected in loss for image: {img_name_str}")
                print(f"Original tensor range: [{original_tensor.min():.5f}, {original_tensor.max():.5f}]")
                print(f"Target image range: [{target_image.min():.5f}, {target_image.max():.5f}]")
                print(f"Transformed image range: [{transformed_image.min():.5f}, {transformed_image.max():.5f}]")
                print(f"Params: {params.detach().cpu().numpy()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

            # Save checkpoint at the middle of each epoch
            if batch_idx == len(train_loader) // 2:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_middle.pth")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / (batch_idx + 1),
                }, checkpoint_path)

        epoch_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed, Loss: {epoch_loss:.4f}")

        scheduler.step(epoch_loss)

        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_end.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)

        # Update the latest checkpoint
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, latest_checkpoint_path)

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