import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat_inplace
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from piq import SSIMLoss

from cfg import Config
from model import Auto_Encoder

class ImagePatchDataset(Dataset):
    def __init__(self, img_dir, patch_size):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img)
        return tensor, tensor

def train():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Auto_Encoder(cfg.patch_size, cfg.latent_size, mode="quant")
    model.train()
    model.qconfig = get_default_qat_qconfig('fbgemm')
    prepare_qat_inplace(model)
    model.to(device)

    dataset = ImagePatchDataset(cfg.data_dir, cfg.patch_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()

    for epoch in range(cfg.epochs):
        progress_bar = tqdm(loader)
        for imgs, _ in progress_bar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            _, output = model(imgs)
            loss = 0.5 * mse_loss(output, imgs) + 0.5 * ssim_loss(output, imgs)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        torch.save(model.state_dict(), f"ckpt/qat_model_final.pth")

if __name__ == '__main__':
    train()
