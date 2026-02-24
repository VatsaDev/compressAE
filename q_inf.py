import os
import sys
import torch
import torch.nn as nn
import numpy as np
import subprocess
from PIL import Image
from torchvision import transforms
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat_inplace, convert

from cfg import Config
from model import Auto_Encoder

def run_inference():
    cfg = Config()
    image_path = "test/hard.jpg"
    checkpoint_path = "ckpt/qat_model_final.pth"
    zip_p = "C:/Program Files/7-Zip/7z.exe" if sys.platform.startswith('win') else "7z"

    model = Auto_Encoder(cfg.patch_size, cfg.latent_size, mode="quant")
    model.eval()
    model.qconfig = get_default_qat_qconfig('fbgemm')
    prepare_qat_inplace(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    int8_model = convert(model.cpu(), inplace=False)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w, new_h = (w // cfg.patch_size) * cfg.patch_size, (h // cfg.patch_size) * cfg.patch_size
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    transform = transforms.ToTensor()
    patches = [transform(img.crop((j, i, j + cfg.patch_size, i + cfg.patch_size))).unsqueeze(0) 
               for i in range(0, new_h, cfg.patch_size) for j in range(0, new_w, cfg.patch_size)]
    patches_tensor = torch.cat(patches, 0)

    with torch.no_grad():
        q_input = int8_model.quant(patches_tensor)
        latent_quantized = int8_model.enc(q_input)
        
    latent_int8 = latent_quantized.int_repr().numpy().astype(np.int8)
    bin_path = "outs/latent.bin"
    latent_int8.tofile(bin_path)
    
    seven_zip_path = "outs/latent.7z"
    subprocess.run([zip_p, 'a', seven_zip_path, bin_path, '-mx=9'])

    print(f"Original Image: {os.path.getsize(image_path)/1024:.2f} KB")
    print(f"INT8 Latent 7z: {os.path.getsize(seven_zip_path)/1024:.2f} KB")

    loaded_int8 = np.fromfile(bin_path, dtype=np.int8).reshape(latent_int8.shape)
    loaded_tensor = torch.from_numpy(loaded_int8).to(torch.uint8) 
    
    scale = latent_quantized.q_scale()
    zero_point = latent_quantized.q_zero_point()
    recon_latent = torch._make_per_tensor_quantized_tensor(loaded_tensor, scale, zero_point)

    with torch.no_grad():
        recon_patches = int8_model.dec(recon_latent)
        recon_patches = int8_model.dequant(recon_patches)

    recon_np = (recon_patches.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    full_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    idx = 0
    for i in range(0, new_h, cfg.patch_size):
        for j in range(0, new_w, cfg.patch_size):
            full_image[i:i+cfg.patch_size, j:j+cfg.patch_size, :] = recon_np[idx]
            idx += 1
    
    Image.fromarray(full_image).save("outs/reconstructed.png")

if __name__ == '__main__':
    run_inference()
