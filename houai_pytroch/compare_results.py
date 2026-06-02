"""
拼接对比脚本：对每张图片分别用 V1 / V2 模型推理，计算各自 Dice 分数，
绿线(真实) + 红线(预测) 叠加，左上角标注 Dice，左右拼接，输出到 result_compare/
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
IM_WIDTH, IM_HEIGHT = 256, 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_V1 = os.path.join(BASE_DIR, 'venv', 'model_pytorch', 'best_model.pth')
MODEL_V2 = os.path.join(BASE_DIR, 'venv', 'model_pytorch_v2', 'best_model.pth')
SRC_DIR = 'D:/Code/code_project/python/project/houai/dataset/srcs'
MASK_DIR = 'D:/Code/code_project/python/project/houai/dataset/labels_png'
OUTPUT_DIR = os.path.join(BASE_DIR, 'result_compare')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 模型定义
# ============================================================
class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super(UNet, self).__init__()
        bn_kwargs = dict(momentum=0.1, eps=0.001)

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

        self.conv5_1 = nn.Conv2d(512, 1024, (3, 3), padding='same')
        self.bn5_1 = nn.BatchNorm2d(1024, **bn_kwargs)
        self.conv5_2 = nn.Conv2d(1024, 1024, (3, 3), padding='same')
        self.bn5_2 = nn.BatchNorm2d(1024, **bn_kwargs)

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

        conv5 = F.relu(self.bn5_1(self.conv5_1(pool4)))
        conv5 = F.relu(self.bn5_2(self.conv5_2(conv5)))

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
# 加载两个模型
# ============================================================
print(f'Loading V1: {MODEL_V1}')
model_v1 = UNet().to(DEVICE)
model_v1.load_state_dict(torch.load(MODEL_V1, map_location=DEVICE, weights_only=True), strict=True)
model_v1.eval()

print(f'Loading V2: {MODEL_V2}')
model_v2 = UNet().to(DEVICE)
model_v2.load_state_dict(torch.load(MODEL_V2, map_location=DEVICE, weights_only=True), strict=True)
model_v2.eval()

print(f'Device: {DEVICE}\n')


# ============================================================
# 推理 + 后处理（与 batch_predict.py 完全一致）
# ============================================================
def predict_mask(model, img_bgr):
    """返回 (pred_mask_uint8, auto_thresh, 叠加后的 BGR 图)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IM_WIDTH, IM_HEIGHT))
    tensor = torch.from_numpy((img_resized / 255.0).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor).cpu().squeeze().numpy()

    # Otsu
    pred_uint8 = (pred * 255).clip(0, 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    auto_thresh = otsu_thresh / 255.0
    if auto_thresh < 0.01:
        auto_thresh = max(pred.max() * 0.5, 0.1)

    # 后处理
    pred_mask = (pred >= auto_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pred_mask = (labels == largest_label).astype(np.uint8) * 255

    return pred_mask


def dice_score(pred_mask, gt_mask):
    """计算 Dice 系数"""
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    intersection = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2 * intersection / denom


def draw_overlay(img_bgr, pred_mask, gt_mask, dice_val, label):
    """在原图上叠加绿(GT)+红(Pred)轮廓，并标注 Dice 和标签"""
    img_out = cv2.resize(img_bgr, (IM_WIDTH, IM_HEIGHT))

    # 预测轮廓（红色）
    contours_p, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_out, contours_p, -1, (0, 0, 255), 1)

    # GT 轮廓（绿色）
    contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_out, contours_gt, -1, (0, 255, 0), 1)

    # 文字标注
    cv2.putText(img_out, f'{label}', (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img_out, f'Dice={dice_val:.4f}', (5, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img_out


# ============================================================
# 主流程
# ============================================================
img_files = sorted(glob(f'{SRC_DIR}/*'))
print(f'Found {len(img_files)} images. Processing...\n')

dice_v1_list = []
dice_v2_list = []
v2_better = 0
v1_better = 0

for img_path in tqdm(img_files):
    name = os.path.basename(img_path)
    mask_path = os.path.join(MASK_DIR, os.path.splitext(name)[0] + '.png')

    # 读取原图
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    # 读取 GT
    if not os.path.exists(mask_path):
        continue
    gt_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt_full, (IM_WIDTH, IM_HEIGHT))
    _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

    # V1 推理
    pm1 = predict_mask(model_v1, img_bgr)
    d1 = dice_score(pm1, gt_bin)
    dice_v1_list.append(d1)

    # V2 推理
    pm2 = predict_mask(model_v2, img_bgr)
    d2 = dice_score(pm2, gt_bin)
    dice_v2_list.append(d2)

    if d2 > d1:
        v2_better += 1
    elif d1 > d2:
        v1_better += 1

    # 画图
    img1 = draw_overlay(img_bgr, pm1, gt_bin, d1, 'V1 (old)')
    img2 = draw_overlay(img_bgr, pm2, gt_bin, d2, 'V2 (new)')

    # 左右拼接
    concat = np.hstack([img1, img2])
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), concat)

# ============================================================
# 汇总
# ============================================================
avg_v1 = np.mean(dice_v1_list) if dice_v1_list else 0
avg_v2 = np.mean(dice_v2_list) if dice_v2_list else 0
tie = len(dice_v1_list) - v2_better - v1_better

summary_lines = [
    f"Per-image Dice 汇总 ({len(dice_v1_list)} images)",
    f"{'='*50}",
    f"  V1 (old)  Avg Dice: {avg_v1:.4f}",
    f"  V2 (new)  Avg Dice: {avg_v2:.4f}",
    f"  V2 > V1:  {v2_better} images",
    f"  V1 > V2:  {v1_better} images",
    f"  Tie:      {tie} images",
    f"{'='*50}",
]

for line in summary_lines:
    print(line)

summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f'\nDone: {len(dice_v1_list)} comparison images + summary.txt saved to {OUTPUT_DIR}')
