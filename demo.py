import sys
import os
# 1. Force Python to look in your NAFNet folder first
sys.path.insert(0, os.path.abspath('.'))

import torch
import cv2
import numpy as np

# 2. The Correct Import Path
from basicsr.models.archs.NAFNet_arch import NAFNetLocal

# --- CONFIGURATION (EDIT THIS) ---
# 1. Path to your best model (The one you just saved)
MODEL_PATH = 'experiments/MixedWeather_RTX4050/models/net_g_22000.pth' 

# 2. The image you want to fix (Put a rainy image in your folder)
INPUT_IMAGE = 'test_image.jpg'  

# 3. Where to save the fixed version
OUTPUT_IMAGE = 'result.jpg'     

# 4. Model Settings (MUST match your train_MixedWeather_RTX4050.yml file!)
# If you used the standard config, these are the defaults:
ENC_BLK_NUMS = [1, 1, 1, 8] 
MIDDLE_BLK_NUM = 1
DEC_BLK_NUMS = [1, 1, 1, 1]
# ---------------------

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Define Model (Make sure this matches your training!)
    net = NAFNetLocal(
        width=32, 
        enc_blk_nums=[1, 1, 1, 8], 
        middle_blk_num=1, 
        dec_blk_nums=[1, 1, 1, 1]
    )
    
    # 2. Load Weights
    checkpoint = torch.load(MODEL_PATH)
    if 'params' in checkpoint:
        net.load_state_dict(checkpoint['params'])
    else:
        net.load_state_dict(checkpoint)
    net.eval()
    net.cuda()

    # 3. Process Image
    print(f"Processing {INPUT_IMAGE}...")
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print("Error: Image not found!")
        return

    # --- NEW: AUTO-RESIZE TRICK ---
    h, w = img.shape[:2]
    print(f"Original Size: {w}x{h}")
    
    # If image is too big, shrink it (Maintain aspect ratio)
    MAX_DIM = 1000
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"Resizing to {new_w}x{new_h} for better detection...")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # ------------------------------

    # Normalize
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        output = net(img)

    # Save
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(OUTPUT_IMAGE, output)
    print(f"Success! Saved result to {OUTPUT_IMAGE}")

if __name__ == '__main__':
    main()