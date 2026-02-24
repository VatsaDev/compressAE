import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from PIL import Image
from tqdm import tqdm
from piq import SSIMLoss
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from cfg import Config
from model import Auto_Encoder

# Ensure directories exist
os.makedirs("ckpt", exist_ok=True)
os.makedirs("plots", exist_ok=True)

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImagePatchDataset(Dataset):
    def __init__(self, img_dir, patch_size):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        try:
            image_rgb = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping corrupted file: {img_path}")
            new_idx = (idx + 1) % len(self.image_files)
            return self.__getitem__(new_idx)
    
        image_tensor = self.transform(image_rgb)
        return image_tensor, image_tensor

dataset = ImagePatchDataset(cfg.data_dir, cfg.patch_size)
train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

def qualitative_test(model, dataset, epoch, device):
    model.eval()
    num_samples = 5
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, _ = dataset[idx]
            img_input = img_tensor.unsqueeze(0).to(device)
            
            _, output = model(img_input)
            
            # Prepare for plotting
            orig_img = img_tensor.permute(1, 2, 0).cpu().numpy()
            recon_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_img = np.clip(recon_img, 0, 1)
            
            # Row 0: Original
            axes[0, i].imshow(orig_img)
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title("Original", loc='left')
            
            # Row 1: Reconstructed
            axes[1, i].imshow(recon_img)
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title("Reconstructed", loc='left')

    plt.suptitle(f"Qualitative Test - Epoch {epoch+1}")
    plt.tight_layout()
    plt.savefig(f"plots/epoch_{epoch+1}_test.png")
    plt.close()
    print(f"Saved qualitative plot to plots/epoch_{epoch+1}_test.png")

# setups 
model = Auto_Encoder(cfg.patch_size, cfg.latent_size, mode="norm").to(device)

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
ssim_loss = SSIMLoss()

optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

def train():
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params/1e6:.2f}M params")
    print(f"\nTraining for {cfg.epochs} epochs in FP32 mode")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for lab_images, _ in progress_bar:
            lab_images = lab_images.to(device)

            optimizer.zero_grad()
            _, output = model(lab_images)

            # Standard FP32 Losses
            mse = mse_loss(output, lab_images)
            mae = mae_loss(output, lab_images)
            ssim = ssim_loss(output, lab_images)
        
            loss = (0.1 * mae + 0.5 * mse + 0.4 * ssim)
    
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'Total': f"{loss.item():.4f}",
                'MSE': f"{mse.item():.4f}",
                'SSIM': f"{ssim.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

        # Qualitative testing
        qualitative_test(model, dataset, epoch, device)

        # Checkpoint save
        save_path = f"ckpt/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    train()
