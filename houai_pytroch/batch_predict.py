"""
批量预测脚本：对所有数据集图片生成 绿线(真实) + 红线(预测) 叠加图，保存到 result/
"""
import os, sys
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
MODEL_PATH = os.path.join(BASE_DIR, 'venv', 'model_pytorch', 'best_model.pth')
SRC_DIR = 'D:/Code/code_project/python/project/houai/dataset/srcs'
MASK_DIR = 'D:/Code/code_project/python/project/houai/dataset/labels_png'
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(RESULT_DIR, exist_ok=True)


def dice_score(pred_mask, gt_mask):
    """计算 Dice 系数"""
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    intersection = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2 * intersection / denom


# ============================================================
# 模型定义（标准 UNet：Conv→BN→ReLU，来自 unet_pytorch）
# ============================================================
class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super(UNet, self).__init__()

        bn_kwargs = dict(momentum=0.1, eps=0.001)

        # Encoder — 标准 Conv→BN→ReLU，每层 Conv 配一个 BN
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

        # Xavier/Glorot uniform 权重初始化（对齐 TF Conv2D 默认值）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder — 标准 Conv→BN→ReLU
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
# 加载模型
# ============================================================
print(f'Loading model: {MODEL_PATH}')
model = UNet().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
print(f'Model loaded. Device: {DEVICE}')


# ============================================================
# 批量预测
# ============================================================
def process_one(img_path):
    """对单张图片进行预测，返回 (原图BGR, 是否成功)"""
    name = os.path.basename(img_path)
    mask_path = os.path.join(MASK_DIR, os.path.splitext(name)[0] + '.png')

    # 读取并预处理
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IM_WIDTH, IM_HEIGHT))
    tensor = torch.from_numpy((img_resized / 255.0).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        pred = model(tensor).cpu().squeeze().numpy()

    # Otsu 自动阈值
    pred_uint8 = (pred * 255).clip(0, 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    auto_thresh = otsu_thresh / 255.0
    if auto_thresh < 0.01:
        auto_thresh = max(pred.max() * 0.5, 0.1)

    # 背景图
    img_bgr = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))

    # ===== 预测轮廓（红色）=====
    pred_mask = (pred >= auto_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    # 只保留最大连通分量
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pred_mask = (labels == largest_label).astype(np.uint8) * 255

    contours_p, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, contours_p, -1, (0, 0, 255), 1)  # 红：预测

    # ===== Ground truth 轮廓（绿色）=====
    if os.path.exists(mask_path):
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (IM_WIDTH, IM_HEIGHT))
        _, thresh_gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours_gt, -1, (0, 255, 0), 1)  # 绿：真实

        # 计算 Dice 并在右上角标注（红色字体）
        try:
            dice_val = dice_score(pred_mask, thresh_gt)
            text = f'Dice={dice_val:.4f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
            x = IM_WIDTH - text_w - 5
            y = 5 + text_h
            cv2.putText(img_bgr, text, (x, y), font, scale, (0, 0, 255), thickness)
        except Exception:
            pass

    return img_bgr


# ============================================================
# 主流程
# ============================================================
img_files = sorted(glob(f'{SRC_DIR}/*'))
print(f'Found {len(img_files)} images. Processing...')

success = 0
for img_path in tqdm(img_files):
    try:
        result = process_one(img_path)
        name = os.path.basename(img_path)
        out_path = os.path.join(RESULT_DIR, name)
        cv2.imwrite(out_path, result)
        success += 1
    except Exception as e:
        print(f'  [ERROR] {os.path.basename(img_path)}: {e}')

print(f'\nDone: {success}/{len(img_files)} images saved to {RESULT_DIR}')
