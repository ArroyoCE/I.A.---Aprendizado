from neural_network import ImprovedStableTransformationParamNet
from SourceImageDataset import SourceImageDataset
from torch.utils.data import DataLoader
from Train_Model_Alternative import train_model
import torch
import os

def main():
    input_dir = "../source_dir"
    target_dir = "../target_dir"
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ImprovedStableTransformationParamNet()
    dataset = SourceImageDataset(input_dir, resize_size=(512, 512))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_model(model, dataloader, optimizer, device=device, checkpoint_dir=checkpoint_dir, target_dir=target_dir)

if __name__ == "__main__":
    main()