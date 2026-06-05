###################################
#####PyTorch
###################################

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found, using CPU instead.")


# 加载图片
###################################################################

# 图片大小
im_width = 256
im_height = 256


train_files = sorted(glob('D:/Code/code_project/python/project/houai/dataset/srcs/*'))
mask_files = sorted(glob('D:/Code/code_project/python/project/houai/dataset/labels_png/*'))

# 打印数据集大小
print(len(train_files))
print(len(mask_files))

########################################################################

#数据可视化
########################################################
#将分割图片和原图片合并到一起显示
# rows,cols=3,3
# fig=plt.figure(figsize=(10,10))

# rows,cols=4,4
# fig=plt.figure(figsize=(16,16))
# for i in range(1,rows*cols+1):
#     fig.add_subplot(rows,cols,i)
#     img_path=train_files[i]
#     msk_path=mask_files[i]
#     img=cv2.imread(img_path)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     msk=cv2.imread(msk_path)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img)
#     plt.imshow(msk,alpha=0.4)
# plt.show()

##########################################################

##划分训练集，实验集，验证集
###############################################################################
df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
df_train, df_test = train_test_split(df,test_size = 0.1)
df_train, df_val = train_test_split(df_train,test_size = 0.2)
#########################输出测试
# print(df_train.values.shape)
# print(df_val.values.shape)
# print(df_test.values.shape)
###########################输出成功
##################################################################


# 定义数据生成器函数
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
        
        # 读取图片和掩码
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, self.target_size)
        
        # 数据增强（对齐 TF ImageDataGenerator 参数）
        if self.augment:
            # 水平翻转 50% 概率（TF horizontal_flip=True）
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            # 仿射变换：旋转、平移、剪切、缩放
            rows, cols = img.shape[:2]
            # 旋转 ±0.2°（TF rotation_range=0.2）
            angle = np.random.uniform(-0.2, 0.2)
            # 平移 5%（TF width/height_shift_range=0.05）
            dx = np.random.uniform(-0.05, 0.05) * cols
            dy = np.random.uniform(-0.05, 0.05) * rows
            # 剪切 0.05 rad（TF shear_range=0.05）
            shear = np.random.uniform(-0.05, 0.05)
            # 缩放 [0.95, 1.05]（TF zoom_range=0.05）
            scale = np.random.uniform(0.95, 1.05)

            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
            M[0, 1] += shear
            M[0, 2] += dx
            M[1, 2] += dy
            img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            mask = cv2.warpAffine(mask, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
        # 调整数据
        img, mask = adjust_data(img, mask)
        
        # 转换为tensor
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return img, mask


# 处理数据函数
def adjust_data(img, mask):
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)
#############################################################################


##指定损失函数以及指标函数
############################################################################
smooth = 1


# 定义Dice系数
def dice_coef(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    And = torch.sum(y_true * y_pred)
    return ((2 * And + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth))


# 定义损失函数：Dice + BCE 组合
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """BCE + Dice 组合损失，两者各占 50%"""
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_coef_loss(y_true, y_pred)  # = -dice, 范围 [-1, 0]
    return 0.5 * bce + 0.5 * (dice + 1.0)  # dice 平移使之为正，便于组合


# 定义iou函数
def iou(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    sum_ = torch.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def jac_distance(y_true, y_pred):
    return -iou(y_true, y_pred)
################################################################################


#######定义U-Net网络
#################################################################################
class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super(UNet, self).__init__()
        
        # Encoder — 标准 Conv→BN→ReLU，每层 Conv 配一个 BN
        self.conv1_1 = nn.Conv2d(input_channels, 64, (3, 3), padding='same')
        self.bn1_1 = nn.BatchNorm2d(64, momentum=0.1, eps=0.001)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bn1_2 = nn.BatchNorm2d(64, momentum=0.1, eps=0.001)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), padding='same')
        self.bn2_1 = nn.BatchNorm2d(128, momentum=0.1, eps=0.001)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn2_2 = nn.BatchNorm2d(128, momentum=0.1, eps=0.001)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), padding='same')
        self.bn3_1 = nn.BatchNorm2d(256, momentum=0.1, eps=0.001)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), padding='same')
        self.bn3_2 = nn.BatchNorm2d(256, momentum=0.1, eps=0.001)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), padding='same')
        self.bn4_1 = nn.BatchNorm2d(512, momentum=0.1, eps=0.001)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), padding='same')
        self.bn4_2 = nn.BatchNorm2d(512, momentum=0.1, eps=0.001)
        self.pool4 = nn.MaxPool2d((2, 2))

        # Bottom
        self.conv5_1 = nn.Conv2d(512, 1024, (3, 3), padding='same')
        self.bn5_1 = nn.BatchNorm2d(1024, momentum=0.1, eps=0.001)
        self.conv5_2 = nn.Conv2d(1024, 1024, (3, 3), padding='same')
        self.bn5_2 = nn.BatchNorm2d(1024, momentum=0.1, eps=0.001)

        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, (2, 2), stride=(2, 2))
        self.conv6_1 = nn.Conv2d(1024, 512, (3, 3), padding='same')
        self.bn6_1 = nn.BatchNorm2d(512, momentum=0.1, eps=0.001)
        self.conv6_2 = nn.Conv2d(512, 512, (3, 3), padding='same')
        self.bn6_2 = nn.BatchNorm2d(512, momentum=0.1, eps=0.001)

        self.up7 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.conv7_1 = nn.Conv2d(512, 256, (3, 3), padding='same')
        self.bn7_1 = nn.BatchNorm2d(256, momentum=0.1, eps=0.001)
        self.conv7_2 = nn.Conv2d(256, 256, (3, 3), padding='same')
        self.bn7_2 = nn.BatchNorm2d(256, momentum=0.1, eps=0.001)

        self.up8 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.conv8_1 = nn.Conv2d(256, 128, (3, 3), padding='same')
        self.bn8_1 = nn.BatchNorm2d(128, momentum=0.1, eps=0.001)
        self.conv8_2 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn8_2 = nn.BatchNorm2d(128, momentum=0.1, eps=0.001)

        self.up9 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.conv9_1 = nn.Conv2d(128, 64, (3, 3), padding='same')
        self.bn9_1 = nn.BatchNorm2d(64, momentum=0.1, eps=0.001)
        self.conv9_2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.bn9_2 = nn.BatchNorm2d(64, momentum=0.1, eps=0.001)
        
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


def unet(input_size=(256, 256, 3)):
    if len(input_size) == 3:
        input_channels = input_size[2]
    else:
        input_channels = 3
    return UNet(input_channels=input_channels)


##设置学习效率
#########################################################################
# EPOCHS = 150
# BATCH_SIZE = 32
##################################
EPOCHS = 250  # 训练轮次
BATCH_SIZE = 6######意味着在每次梯度更新时，模型会使用6个样本组成一个批次来计算梯度并更新参数 。
####################################

import math

def lrfn(epoch):
    LR_MIN = 0.00001
    LR_MAX = 0.001
    LR_RAMPUP_EPOCHS = 10       # 10 轮 warmup
    LR_COSINE_EPOCHS = EPOCHS - LR_RAMPUP_EPOCHS  # 90 轮余弦退火

    if epoch < LR_RAMPUP_EPOCHS:
        # 线性 warmup：从 LR_MIN 到 LR_MAX
        lr = LR_MIN + (LR_MAX - LR_MIN) * epoch / LR_RAMPUP_EPOCHS
    else:
        # 余弦退火：从 LR_MAX 平滑衰减到 LR_MIN
        progress = (epoch - LR_RAMPUP_EPOCHS) / LR_COSINE_EPOCHS
        progress = min(progress, 1.0)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
    return lr


rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
###########################################################################################


#########开始训练
######################################################################
# 创建数据集
train_dataset = UNetDataset(df_train, target_size=(im_height, im_width), augment=True)
val_dataset = UNetDataset(df_val, target_size=(im_height, im_width), augment=False)
test_dataset = UNetDataset(df_test, target_size=(im_height, im_width), augment=False)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = unet(input_size=(im_height, im_width, 3))
model = model.to(device)

os.makedirs('D:/Code/code_project/python/project/houai/houai_pytroch/model_pytorch', exist_ok=True)

# 定义优化器和学习率调度器
# LambdaLR 实际 lr = base_lr * lr_lambda(epoch)，设 base_lr=1.0 使 lrfn 直接等于实际学习率（对齐 TF）
optimizer = optim.Adam(model.parameters(), lr=1.0)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrfn)

# 训练历史记录
history = {
    'loss': [],
    'val_loss': [],
    'dice_coef': [],
    'val_dice_coef': [],
    'iou': [],
    'val_iou': [],
    'binary_accuracy': [],
    'val_binary_accuracy': []
}

best_val_loss = float('inf')

# 开始训练
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_iou_val = 0.0
    train_acc = 0.0
    
    for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        img, mask = img.to(device), mask.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        
        loss = combined_loss(mask, output)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_dice += dice_coef(mask, output).item()
        train_iou_val += iou(mask, output).item()
        
        # 计算准确率
        pred_binary = (output > 0.5).float()
        accuracy = torch.mean((pred_binary == mask).float()).item()
        train_acc += accuracy
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou_val = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            
            loss = combined_loss(mask, output)
            val_loss += loss.item()
            val_dice += dice_coef(mask, output).item()
            val_iou_val += iou(mask, output).item()
            
            # 计算准确率
            pred_binary = (output > 0.5).float()
            accuracy = torch.mean((pred_binary == mask).float()).item()
            val_acc += accuracy
    
    # 计算平均值
    train_loss /= len(train_loader)
    train_dice /= len(train_loader)
    train_iou_val /= len(train_loader)
    train_acc /= len(train_loader)
    
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou_val /= len(val_loader)
    val_acc /= len(val_loader)
    
    # 记录历史
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['dice_coef'].append(train_dice)
    history['val_dice_coef'].append(val_dice)
    history['iou'].append(train_iou_val)
    history['val_iou'].append(val_iou_val)
    history['binary_accuracy'].append(train_acc)
    history['val_binary_accuracy'].append(val_acc)
    
    # 打印信息
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
    print(f"  Train IOU: {train_iou_val:.4f}, Val IOU: {val_iou_val:.4f}")
    print(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # 保存验证集 loss 最低的模型（最佳模型）
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'D:/Code/code_project/python/project/houai/houai_pytroch/model_pytorch/best_model.pth')
        print(f'  >>> Best model saved (val_loss={best_val_loss:.4f})')

    # 学习率调度
    scheduler.step()

############################################################# ##############################


##########################################保存最终模型
torch.save(model.state_dict(), 'D:/Code/code_project/python/project/houai/houai_pytroch/model_pytorch/final_model.pth')  # 训练结束时的模型
print(f'Final model saved.')
###################################################


#########绘制损失值和准确率曲线

# ==========================================
# Figure: 训练 vs 验证 对比（Loss + Dice 同框）
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['loss'], 'r-', label='Train Loss', linewidth=1.5)
ax1.plot(history['val_loss'], 'b-', label='Val Loss', linewidth=1.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss — Train vs Validation', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['dice_coef'], 'r-', label='Train Dice', linewidth=1.5)
ax2.plot(history['val_dice_coef'], 'b-', label='Val Dice', linewidth=1.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Dice Coefficient')
ax2.set_title('Dice — Train vs Validation', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle('Train vs Validation Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
from datetime import datetime
comparison_name = f"D:/Code/code_project/python/project/houai/houai_pytroch/model_pytorch/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(comparison_name, dpi=150)
print(f'Plot saved: {comparison_name}')
plt.close('all')
print('Plot saved: comparison.png')
# #########################################################


# #####测试集准确率
# ########################################
model.eval()
test_loss = 0.0
test_iou_val = 0.0
test_dice = 0.0

# 逐样本记录，用于折线图
sample_losses = []
sample_ious = []
sample_dices = []

with torch.no_grad():
    for img, mask in test_loader:
        img, mask = img.to(device), mask.to(device)
        output = model(img)

        # 逐样本计算
        for i in range(img.size(0)):
            single_loss = dice_coef_loss(mask[i:i+1], output[i:i+1]).item()
            single_iou = iou(mask[i:i+1], output[i:i+1]).item()
            single_dice = dice_coef(mask[i:i+1], output[i:i+1]).item()
            sample_losses.append(-single_loss)  # 取反，值越大越好
            sample_ious.append(single_iou)
            sample_dices.append(single_dice)

        test_loss += dice_coef_loss(mask, output).item()
        test_iou_val += iou(mask, output).item()
        test_dice += dice_coef(mask, output).item()

test_loss /= len(test_loader)
test_iou_val /= len(test_loader)
test_dice /= len(test_loader)

print("Test loss: ", test_loss)
print("Test IOU: ", test_iou_val)
print("Test Dice Coefficient: ", test_dice)

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'D:/Code/code_project/python/project/houai/houai_pytroch/model_pytorch/test_results_{timestamp_str}.txt', 'w') as f:
    f.write(f"Test loss: {test_loss}\n")
    f.write(f"Test IOU: {test_iou_val}\n")
    f.write(f"Test Dice Coefficient: {test_dice}\n")

# ==========================================
# Figure: 测试集 — 逐样本 Dice / IOU 折线图
# ==========================================
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8))

x = range(len(sample_dices))

ax3.plot(x, sample_dices, 'g-', linewidth=1, alpha=0.7, label='Dice (per sample)')
ax3.axhline(y=test_dice, color='r', linestyle='--', linewidth=1.5, label=f'Avg Dice = {test_dice:.4f}')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Dice Coefficient')
ax3.set_title('Test Set — Per-Sample Dice Coefficient', fontsize=13)
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

fig2.suptitle('Test Set Per-Sample Metrics', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/Code/code_project/python/project/houai/houai_pytroch/model_pytorch/test_metrics.png', dpi=150)
plt.close('all')

print('Plots saved: comparison.png, test_metrics.png')

# ==========================================
# BN 统计量重新校准（修复 eval 模式输出异常）
# ==========================================
print('Recalibrating BN running stats...')
model.train()  # train 模式：BN 用 batch 统计更新 running stats
with torch.no_grad():
    for img, _ in tqdm(train_loader, desc='BN recalibration'):
        img = img.to(device)
        model(img)  # 仅前向传播，更新 BN running_mean/running_var
torch.save(model.state_dict(), 'D:/Code/code_project/python/project/houai/houai_pytroch/model_pytorch/best_model.pth')
print('BN recalibrated model saved to best_model.pth')
# #########################################################
