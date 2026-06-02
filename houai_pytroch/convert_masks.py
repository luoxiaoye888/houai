"""将 JPG 掩码转换为 PNG 格式，消除压缩伪影"""
import cv2
import numpy as np
from glob import glob
import os

BASE = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(BASE, '..', 'dataset', 'labels_show')
dst_dir = os.path.join(BASE, '..', 'dataset', 'labels_png')
os.makedirs(dst_dir, exist_ok=True)

for f in sorted(glob(f'{src_dir}/*')):
    mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    # 二值化：原 JPG 中值在 0-7 / 248-255，用 127 阈值还原为 0/255
    mask_bin = (mask > 127).astype(np.uint8) * 255
    name = os.path.splitext(os.path.basename(f))[0] + '.png'
    cv2.imwrite(os.path.join(dst_dir, name), mask_bin)
    if len(glob(f'{dst_dir}/*')) <= 5:
        print(f'  {f} -> {name}')

print(f'Done: {len(glob(f"{dst_dir}/*"))} masks converted to PNG in {dst_dir}')
