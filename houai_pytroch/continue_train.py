"""
继续训练脚本：加载 best_model.pth，用新 LR 策略继续训练
不动原始 unet_pytorch.py，所有产出输出到 model_pytorch_v2/
"""
import os
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================================================
# 配置
# ============================================================
IM_WIDTH, IM_HEIGHT = 256, 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'venv', 'model_pytorch', 'best_model.pth')
OUTPUT_DIR = os.path.join(BASE_DIR, 'venv', 'model_pytorch_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练参数
ADDITIONAL_EPOCHS = 100   # 额外训练轮次
BATCH_SIZE = 6
LR_START = 5e-4           # 从 best model 继续，不需要 1e-3 那么大
LR_MIN = 3e-5             # 提高最低 LR，避免后期冻结（原来 1e-5 太小）
SEED = 42                 # 固定随机种子，保证可复现

# ============================================================
# 数据增强参数（可单独调整）
# ============================================================
AUG_GEOMETRIC = True       # 几何增强：旋转/平移/剪切/缩放（同时作用于 img 和 mask）
AUG_COLOR = True           # 色彩增强：亮度/对比度/伽马（仅作用于 img）
AUG_NOISE_BLUR = True      # 噪声/模糊增强（仅作用于 img）

# 几何增强参数（增强了原版 ±0.2° / 5% / 0.05 / 0.95-1.05 的极保守设置）
ROTATION_DEG = 20          # 旋转范围 ±20°（原来是 ±0.2°，几乎不转）
SHIFT_FRAC = 0.15          # 平移比例 ±15%（原来是 ±5%）
SHEAR_DEG = 10             # 剪切 ±10°（原来是 0.05 rad ≈ 2.9°）
ZOOM_RANGE = (0.85, 1.15)  # 缩放范围 85%~115%（原来是 95%~105%）
FLIP_H_PROB = 0.5          # 水平翻转概率
FLIP_V_PROB = 0.2          # 垂直翻转概率（病灶方向通常有特定朝向，设低一些）

# 色彩增强参数
BRIGHTNESS_RANGE = 0.15    # 亮度调整 ±15%
CONTRAST_RANGE = 0.15      # 对比度调整 ±15%
GAMMA_RANGE = (0.7, 1.5)   # 伽马校正范围

# 噪声/模糊增强参数
GAUSS_NOISE_STD = 0.02     # 高斯噪声标准差（img 已归一化到 [0,1]）
GAUSS_BLUR_SIGMA = 1.0     # 高斯模糊 sigma

# ============================================================
# 固定随机种子
# ============================================================
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 模型定义（与 unet_pytorch.py 完全一致）
# ============================================================
class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super(UNet, self).__init__()

        bn_kwargs = dict(momentum=0.1, eps=0.001)

        # Encoder
        self.conv1_1 = nn.Conv2d(input_channels, 64, (3, 3), padding='same')
        self.bn1_1 = nn.BatchNorm2d(64, **bn_kwargs)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bn1_2 = nn.BatchNorm2d(64, **bn_kwargs)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), padding='same')
        self.bn2_1 = nn.BatchNorm2d(128, **bn_kwargs)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn2_2 = nn.BatchNorm2d(128, **bn_kwargs)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), padding='same')
        self.bn3_1 = nn.BatchNorm2d(256, **bn_kwargs)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), padding='same')
        self.bn3_2 = nn.BatchNorm2d(256, **bn_kwargs)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), padding='same')
        self.bn4_1 = nn.BatchNorm2d(512, **bn_kwargs)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), padding='same')
        self.bn4_2 = nn.BatchNorm2d(512, **bn_kwargs)
        self.pool4 = nn.MaxPool2d((2, 2))

        # Bottom
        self.conv5_1 = nn.Conv2d(512, 1024, (3, 3), padding='same')
        self.bn5_1 = nn.BatchNorm2d(1024, **bn_kwargs)
        self.conv5_2 = nn.Conv2d(1024, 1024, (3, 3), padding='same')
        self.bn5_2 = nn.BatchNorm2d(1024, **bn_kwargs)

        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, (2, 2), stride=(2, 2))
        self.conv6_1 = nn.Conv2d(1024, 512, (3, 3), padding='same')
        self.bn6_1 = nn.BatchNorm2d(512, **bn_kwargs)
        self.conv6_2 = nn.Conv2d(512, 512, (3, 3), padding='same')
        self.bn6_2 = nn.BatchNorm2d(512, **bn_kwargs)

        self.up7 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.conv7_1 = nn.Conv2d(512, 256, (3, 3), padding='same')
        self.bn7_1 = nn.BatchNorm2d(256, **bn_kwargs)
        self.conv7_2 = nn.Conv2d(256, 256, (3, 3), padding='same')
        self.bn7_2 = nn.BatchNorm2d(256, **bn_kwargs)

        self.up8 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.conv8_1 = nn.Conv2d(256, 128, (3, 3), padding='same')
        self.bn8_1 = nn.BatchNorm2d(128, **bn_kwargs)
        self.conv8_2 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn8_2 = nn.BatchNorm2d(128, **bn_kwargs)

        self.up9 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.conv9_1 = nn.Conv2d(128, 64, (3, 3), padding='same')
        self.bn9_1 = nn.BatchNorm2d(64, **bn_kwargs)
        self.conv9_2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bn9_2 = nn.BatchNorm2d(64, **bn_kwargs)

        self.conv10 = nn.Conv2d(64, 1, (1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.bn1_1(self.conv1_1(x)))
        conv1 = F.relu(self.bn1_2(self.conv1_2(conv1)))
        pool1 = self.pool1(conv1)

        conv2 = F.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2 = F.relu(self.bn2_2(self.conv2_2(conv2)))
        pool2 = self.pool2(conv2)

        conv3 = F.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3 = F.relu(self.bn3_2(self.conv3_2(conv3)))
        pool3 = self.pool3(conv3)

        conv4 = F.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4 = F.relu(self.bn4_2(self.conv4_2(conv4)))
        pool4 = self.pool4(conv4)

        # Bottom
        conv5 = F.relu(self.bn5_1(self.conv5_1(pool4)))
        conv5 = F.relu(self.bn5_2(self.conv5_2(conv5)))

        # Decoder
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = F.relu(self.bn6_1(self.conv6_1(up6)))
        conv6 = F.relu(self.bn6_2(self.conv6_2(conv6)))

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = F.relu(self.bn7_1(self.conv7_1(up7)))
        conv7 = F.relu(self.bn7_2(self.conv7_2(conv7)))

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = F.relu(self.bn8_1(self.conv8_1(up8)))
        conv8 = F.relu(self.bn8_2(self.conv8_2(conv8)))

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = F.relu(self.bn9_1(self.conv9_1(up9)))
        conv9 = F.relu(self.bn9_2(self.conv9_2(conv9)))

        return torch.sigmoid(self.conv10(conv9))


# ============================================================
# 数据集
# ============================================================
train_files = sorted(glob('D:/Code/code_project/python/project/houai/dataset/srcs/*'))
mask_files = sorted(glob('D:/Code/code_project/python/project/houai/dataset/labels_png/*'))
print(f'Images: {len(train_files)}, Masks: {len(mask_files)}')

df = pd.DataFrame(data={"filename": train_files, 'mask': mask_files})
df_train, df_test = train_test_split(df, test_size=0.1, random_state=SEED)
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=SEED)
print(f'Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}')


def adjust_data(img, mask):
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)


class UNetDataset(Dataset):
    def __init__(self, data_frame, target_size=(256, 256), augment=False):
        self.data_frame = data_frame
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame['filename'].iloc[idx]
        mask_path = self.data_frame['mask'].iloc[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, self.target_size)

        if self.augment:
            # ============================================================
            # 1. 翻转（几何：img + mask 同步）
            # ============================================================
            if np.random.rand() < FLIP_H_PROB:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if np.random.rand() < FLIP_V_PROB:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)

            # ============================================================
            # 2. 仿射变换（几何：img + mask 同步）
            #    旋转 + 平移 + 剪切 + 缩放，合成一个变换矩阵
            # ============================================================
            if AUG_GEOMETRIC:
                rows, cols = img.shape[:2]

                angle = np.random.uniform(-ROTATION_DEG, ROTATION_DEG)
                dx = np.random.uniform(-SHIFT_FRAC, SHIFT_FRAC) * cols
                dy = np.random.uniform(-SHIFT_FRAC, SHIFT_FRAC) * rows
                shear = np.random.uniform(-SHEAR_DEG, SHEAR_DEG) * math.pi / 180.0
                scale = np.random.uniform(*ZOOM_RANGE)

                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
                M[0, 1] += shear
                M[0, 2] += dx
                M[1, 2] += dy
                img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
                mask = cv2.warpAffine(mask, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

            # ============================================================
            # 3. 色彩增强（仅 img，不改变 mask）
            # ============================================================
            if AUG_COLOR:
                # 亮度调整
                if np.random.rand() < 0.5:
                    delta = np.random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
                    img = (img.astype(np.float32) / 255.0 + delta).clip(0, 1)
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                # 对比度调整（RGB → YUV，调 V 通道）
                if np.random.rand() < 0.5:
                    alpha = 1.0 + np.random.uniform(-CONTRAST_RANGE, CONTRAST_RANGE)
                    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                    img_yuv[:, :, 0] = (img_yuv[:, :, 0].astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
                    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
                else:
                    img = img.astype(np.uint8)

                # 伽马校正
                if np.random.rand() < 0.5:
                    gamma = np.random.uniform(*GAMMA_RANGE)
                    img_float = img.astype(np.float32) / 255.0
                    img = (np.power(img_float, gamma) * 255).clip(0, 255).astype(np.uint8)

            # ============================================================
            # 4. 噪声/模糊增强（仅 img，不改变 mask）
            #    模拟不同成像质量
            # ============================================================
            if AUG_NOISE_BLUR:
                # 高斯噪声
                if np.random.rand() < 0.3:
                    img_float = img.astype(np.float32) / 255.0
                    noise = np.random.randn(*img_float.shape).astype(np.float32) * GAUSS_NOISE_STD
                    img = (img_float + noise).clip(0, 1)
                    img = (img * 255).astype(np.uint8)

                # 高斯模糊
                if np.random.rand() < 0.3:
                    ksize = np.random.choice([3, 5])
                    img = cv2.GaussianBlur(img, (ksize, ksize), GAUSS_BLUR_SIGMA)

        img, mask = adjust_data(img, mask)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return img, mask


# ============================================================
# 损失函数 & 指标（与原版一致）
# ============================================================
smooth = 1


def dice_coef(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    And = torch.sum(y_true * y_pred)
    return ((2 * And + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_coef_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * (dice + 1.0)


def iou(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    sum_ = torch.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


# ============================================================
# 加载 best_model
# ============================================================
print(f'\nLoading best model: {BEST_MODEL_PATH}')
model = UNet().to(DEVICE)
state = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
print('Model loaded successfully.')

# ============================================================
# LR 策略：余弦退火（不需要 warmup，从 best model 直接余弦）
# ============================================================
def lrfn(epoch):
    """余弦退火：LR_START → LR_MIN，无 warmup"""
    progress = epoch / ADDITIONAL_EPOCHS
    progress = min(progress, 1.0)
    lr = LR_MIN + 0.5 * (LR_START - LR_MIN) * (1 + math.cos(math.pi * progress))
    return lr

rng = [i for i in range(ADDITIONAL_EPOCHS)]
y = [lrfn(x) for x in rng]
plt.figure(figsize=(8, 3))
plt.plot(rng, y)
plt.xlabel('Epoch (additional)')
plt.ylabel('Learning Rate')
plt.title(f'Continue Training LR Schedule: {LR_START:.1e} → {LR_MIN:.1e}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lr_schedule.png'), dpi=100)
plt.close()
print(f'LR schedule: {y[0]:.3g} → {y[-1]:.3g}')

# ============================================================
# 数据加载器
# ============================================================
train_dataset = UNetDataset(df_train, target_size=(IM_HEIGHT, IM_WIDTH), augment=True)
val_dataset = UNetDataset(df_val, target_size=(IM_HEIGHT, IM_WIDTH), augment=False)
test_dataset = UNetDataset(df_test, target_size=(IM_HEIGHT, IM_WIDTH), augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 优化器 & 调度器
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=1.0)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrfn)

# ============================================================
# 继续训练
# ============================================================
history = {
    'loss': [], 'val_loss': [],
    'dice_coef': [], 'val_dice_coef': [],
    'iou': [], 'val_iou': [],
}

best_val_loss = float('inf')
best_epoch = 0

print(f'\n{"="*60}')
print(f'Continue training: {ADDITIONAL_EPOCHS} epochs')
print(f'LR: {LR_START:.1e} → {LR_MIN:.1e} (cosine, no warmup)')
print(f'Batch size: {BATCH_SIZE}')
print(f'{"="*60}\n')

for epoch in range(ADDITIONAL_EPOCHS):
    # ---- Train ----
    model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_iou_val = 0.0

    for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{ADDITIONAL_EPOCHS}"):
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        output = model(img)
        loss = combined_loss(mask, output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_coef(mask, output).item()
        train_iou_val += iou(mask, output).item()

    # ---- Val ----
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou_val = 0.0

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            output = model(img)
            loss = combined_loss(mask, output)
            val_loss += loss.item()
            val_dice += dice_coef(mask, output).item()
            val_iou_val += iou(mask, output).item()

    # ---- 平均 ----
    train_loss /= len(train_loader)
    train_dice /= len(train_loader)
    train_iou_val /= len(train_loader)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou_val /= len(val_loader)

    # ---- 记录 ----
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['dice_coef'].append(train_dice)
    history['val_dice_coef'].append(val_dice)
    history['iou'].append(train_iou_val)
    history['val_iou'].append(val_iou_val)

    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1}/{ADDITIONAL_EPOCHS} | LR: {current_lr:.2e}")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
    print(f"  Train IOU:  {train_iou_val:.4f}, Val IOU:  {val_iou_val:.4f}")

    # ---- 保存最佳 ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f'  >>> Best model saved (val_loss={best_val_loss:.4f})')

    scheduler.step()

# ============================================================
# 保存最终模型
# ============================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
print(f'\nFinal model saved. Best epoch: {best_epoch}/{ADDITIONAL_EPOCHS}')

# ============================================================
# BN 重校准 + 保存
# ============================================================
print('\nRecalibrating BN running stats...')
model.train()
with torch.no_grad():
    for img, _ in tqdm(train_loader, desc='BN recalibration'):
        img = img.to(DEVICE)
        model(img)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
print('BN recalibrated model saved.')

# ============================================================
# 测试集评估
# ============================================================
print(f'\n{"="*60}')
print('Test set evaluation')
print(f'{"="*60}')

model.eval()
test_loss = 0.0
test_iou_val = 0.0
test_dice = 0.0

sample_losses = []
sample_ious = []
sample_dices = []

with torch.no_grad():
    for img, mask in test_loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        output = model(img)

        for i in range(img.size(0)):
            single_loss = dice_coef_loss(mask[i:i+1], output[i:i+1]).item()
            single_iou = iou(mask[i:i+1], output[i:i+1]).item()
            single_dice = dice_coef(mask[i:i+1], output[i:i+1]).item()
            sample_losses.append(-single_loss)
            sample_ious.append(single_iou)
            sample_dices.append(single_dice)

        test_loss += dice_coef_loss(mask, output).item()
        test_iou_val += iou(mask, output).item()
        test_dice += dice_coef(mask, output).item()

test_loss /= len(test_loader)
test_iou_val /= len(test_loader)
test_dice /= len(test_loader)

print(f"Test Dice: {test_dice:.4f}")
print(f"Test IOU:  {test_iou_val:.4f}")
print(f"Test Loss: {test_loss:.4f}")

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(os.path.join(OUTPUT_DIR, f'test_results_{timestamp_str}.txt'), 'w') as f:
    f.write(f"Test loss: {test_loss}\n")
    f.write(f"Test IOU: {test_iou_val}\n")
    f.write(f"Test Dice Coefficient: {test_dice}\n")
    f.write(f"Best epoch: {best_epoch}/{ADDITIONAL_EPOCHS}\n")

# ============================================================
# 训练曲线 — Loss + Dice
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['loss'], 'r-', label='Train Loss', linewidth=1.5)
ax1.plot(history['val_loss'], 'b-', label='Val Loss', linewidth=1.5)
ax1.set_xlabel('Epoch (additional)')
ax1.set_ylabel('Loss')
ax1.set_title('Continue Training — Loss', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['dice_coef'], 'r-', label='Train Dice', linewidth=1.5)
ax2.plot(history['val_dice_coef'], 'b-', label='Val Dice', linewidth=1.5)
ax2.axhline(y=0.7143, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Old best Dice (0.7143)')
ax2.set_xlabel('Epoch (additional)')
ax2.set_ylabel('Dice Coefficient')
ax2.set_title('Continue Training — Dice', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(f'Continue Training (additional {ADDITIONAL_EPOCHS} epochs)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
plt.close()

# ============================================================
# 测试集逐样本分布
# ============================================================
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8))

x = range(len(sample_dices))
ax3.plot(x, sample_dices, 'g-', linewidth=1, alpha=0.7, label='Dice (per sample)')
ax3.axhline(y=test_dice, color='r', linestyle='--', linewidth=1.5, label=f'Avg Dice = {test_dice:.4f}')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Dice Coefficient')
ax3.set_title('Test Set — Per-Sample Dice', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1.05)

ax4.plot(x, sample_ious, 'orange', linewidth=1, alpha=0.7, label='IOU (per sample)')
ax4.axhline(y=test_iou_val, color='r', linestyle='--', linewidth=1.5, label=f'Avg IOU = {test_iou_val:.4f}')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('IOU (Jaccard)')
ax4.set_title('Test Set — Per-Sample IOU', fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.05)

fig2.suptitle(f'Test Set Per-Sample Metrics (after +{ADDITIONAL_EPOCHS} epochs)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'test_metrics.png'), dpi=150)
plt.close()

print(f'\nAll outputs saved to {OUTPUT_DIR}/')
print('Done!')
