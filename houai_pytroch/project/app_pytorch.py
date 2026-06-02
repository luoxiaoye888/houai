from flask import Flask, jsonify, request
import os
import sys
import traceback
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务端无 GUI，必须用 Agg 后端，否则 plt 直接崩溃
import matplotlib.pyplot as plt
import base64
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading

# matplotlib 在多线程环境下非线程安全，加锁保护
_plt_lock = threading.Lock()


# ============================================================
# 模型定义（与 unet_pytorch.py 完全一致）
# ============================================================
class UNet(nn.Module):
    """U-Net 分割网络 —— 与 unet_pytorch.py 完全一致"""
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

        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot uniform（对齐 TF Conv2D 默认值）"""
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


# ============================================================
# 配置
# ============================================================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# CORS 支持 —— HTML 页面通过 file:// 或不同端口访问时，浏览器要求跨域
@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return resp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'venv', 'model_pytorch')
PIC_DIR = os.path.join(BASE_DIR, 'static', 'pic')
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, '..', '..', 'dataset', 'labels_png')

IM_WIDTH = 256
IM_HEIGHT = 256
THRESHOLD = 0.3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型缓存: {模型文件名: model}
_model_cache: dict = {}


# ============================================================
# 模型加载（按需加载并缓存）
# ============================================================
def get_model(model_name: str):
    """加载指定的模型文件，已加载的会缓存复用"""
    if model_name not in _model_cache:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')

        m = unet(input_size=(IM_HEIGHT, IM_WIDTH, 3)).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        m.load_state_dict(state_dict, strict=True)
        # 保持 train 模式：BN 用当前图片自身统计量归一化
        # torch.no_grad() 已保证权重不会被修改
        _model_cache[model_name] = m

        # 诊断：打印关键层权重统计，验证模型是否被正确训练
        print(f'Model loaded: {model_name}  ({model_path})')
        for name in ['conv1_1.weight', 'conv5_2.weight', 'conv10.weight',
                      'bn1.running_mean', 'bn5.running_mean']:
            w = state_dict[name]
            print(f'  {name}: mean={w.mean():.6f}, std={w.std():.6f}, min={w.min():.6f}, max={w.max():.6f}')

        # 自检：用一张训练集图片跑推理，看输出范围
        _self_test(m)

    return _model_cache[model_name]


def _self_test(m):
    """加载一张训练集图片，打印模型输出范围，验证推理链路"""
    from glob import glob
    test_imgs = glob('D:/Code/code_project/python/project/houai/dataset/srcs/*')
    if not test_imgs:
        return
    img = cv2.imread(test_imgs[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
    img = img / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = m(tensor).cpu().squeeze().numpy()
    print(f'[SELF-TEST] pred min={pred.min():.6f}, max={pred.max():.6f}, '
          f'mean={pred.mean():.6f}, nnz>=0.1={int((pred>=0.1).sum())}')


# 默认模型
DEFAULT_MODEL = 'best_model.pth'

# 启动时预加载默认模型
try:
    get_model(DEFAULT_MODEL)
    print(f'Device: {DEVICE}')
except FileNotFoundError as e:
    print(f'Warning: {e}')


# ============================================================
# 预处理（与 xin.py Dataset.__getitem__ 完全一致）
# ============================================================
def decode_and_preprocess(image_bytes: bytes) -> torch.Tensor:
    """
    base64 图片字节 → (1, 3, 256, 256) tensor.
    与 unet_pytorch.py adjust_data 完全一致：cv2 解码 → BGR2RGB → resize → /255 → CHW
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
    img = img / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)  # HWC → CHW
    return tensor.to(DEVICE)


# ============================================================
# 预测核心逻辑
# ============================================================
def predict_houai(image_bytes: bytes, img_name: str, model) -> str:
    """
    对输入图像进行病灶分割预测，并与 ground truth 叠加对比。
    返回保存的图片文件名。
    """
    # 预处理（cv2 解码，与训练完全一致）
    input_tensor = decode_and_preprocess(image_bytes)  # (1, 3, 384, 384)

    # 推理 — UNet 输出已含 sigmoid，直接取 [0,1] 概率
    with torch.no_grad():
        pred = model(input_tensor).cpu().squeeze().numpy()

    # 诊断
    print(f'[DEBUG] pred min={pred.min():.6f}, max={pred.max():.6f}, '
          f'mean={pred.mean():.6f}')

    # 自动阈值：Otsu 大津法
    pred_uint8 = (pred * 255).clip(0, 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    auto_thresh = otsu_thresh / 255.0
    if auto_thresh < 0.01:  # Otsu 失败（全黑），回退到 0.5 * max
        auto_thresh = max(pred.max() * 0.5, 0.1)
    print(f'[DEBUG] auto_threshold={auto_thresh:.4f} (Otsu={otsu_thresh})')

    # 背景图（cv2 解码，与训练一致）
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (IM_HEIGHT, IM_WIDTH))

    # 预测 → 二值化（自动阈值）→ 后处理 → 轮廓
    pred_mask = (pred >= auto_thresh).astype(np.uint8) * 255

    # 形态学闭运算：填补碎片间的小缝隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    # 只保留最大连通分量，去掉噪声碎片
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    if num_labels > 1:
        # stats[0] 是背景，跳过；找面积最大的
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pred_mask = (labels == largest_label).astype(np.uint8) * 255

    contours_p, _ = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'[DEBUG] prediction contours found: {len(contours_p)}')

    # ground truth → 轮廓（保持原始扩展名，labels_show 中文件与 srcs 同名同后缀）
    gt_name = os.path.splitext(img_name)[0] + '.png'  # labels_png 统一为 .png
    gt_path = os.path.join(GROUND_TRUTH_DIR, gt_name)
    if os.path.exists(gt_path):
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.resize(ground_truth, (IM_HEIGHT, IM_WIDTH))
        _, thresh_gt = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
        contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours_gt, 0, (0, 255, 0), 1)  # 绿: ground truth
        print(f'[DEBUG] gt contours found: {len(contours_gt)}')

    cv2.drawContours(img_bgr, contours_p, 0, (0, 0, 255), 1)  # 红: 预测

    # 保存结果（matplotlib 用 RGB）
    os.makedirs(PIC_DIR, exist_ok=True)
    pic_name = f"Predict_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    pic_path = os.path.join(PIC_DIR, pic_name)

    with _plt_lock:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(pic_path, dpi=64, bbox_inches='tight')
        plt.close()

    return pic_name


# ============================================================
# 路由
# ============================================================
@app.route('/')
def index():
    return jsonify({'message': 'PyTorch throat cancer segmentation API', 'device': str(DEVICE)}), 200


@app.route('/models', methods=['GET'])
def list_models():
    """返回 venv/model/ 下所有可用的 .pth 模型文件"""
    models = []
    if os.path.isdir(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith('.pth') or f.endswith('.pt'):
                models.append(f)
    return jsonify({'models': sorted(models), 'default': DEFAULT_MODEL}), 200


@app.route('/predict_houai', methods=['POST'])
def predict_houai_api():
    try:
        data = request.get_json(force=True)
        image_b64 = data['image']
        img_name = data['imgName']
        model_name = data.get('model', DEFAULT_MODEL)

        image_bytes = base64.b64decode(image_b64)

        try:
            m = get_model(model_name)
        except FileNotFoundError as e:
            return jsonify({'message': str(e)}), 400

        result_img = predict_houai(image_bytes, img_name, m)

        return jsonify({'result_img': result_img, 'model': model_name}), 200

    except Exception as e:
        print(f'[ERROR] predict_houai failed:', file=sys.stderr)
        traceback.print_exc()
        return jsonify({'message': f'{type(e).__name__}: {e}'}), 500


@app.route('/test_predict', methods=['GET'])
def test_predict():
    """用训练集图片直接测试模型，绕过前端，验证模型本身是否正常"""
    from glob import glob
    train_imgs = glob('D:/Code/code_project/python/project/houai/dataset/srcs/*')
    if not train_imgs:
        return jsonify({'message': '训练集图片未找到'}), 404

    test_img_path = train_imgs[0]
    test_mask_path = test_img_path.replace('srcs', 'labels_png').replace('.jpg', '.png')
    print(f'[TEST] Using image: {test_img_path}')
    print(f'[TEST] Using mask: {test_mask_path}')

    # 与 unet_pytorch.py adjust_data 完全一致
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
    img_input = img / 255.0
    tensor = torch.from_numpy(img_input.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    m = get_model(DEFAULT_MODEL)

    # 测试 eval 模式 — UNet 输出已含 sigmoid
    with torch.no_grad():
        pred = m(tensor).cpu().squeeze().numpy()

    print(f'[TEST] eval mode: min={pred.min():.6f}, max={pred.max():.6f}, '
          f'mean={pred.mean():.6f}')

    # Otsu 自动阈值
    pred_uint8 = (pred * 255).clip(0, 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    auto_thresh = otsu_thresh / 255.0
    if auto_thresh < 0.01:
        auto_thresh = max(pred.max() * 0.5, 0.1)
    print(f'[TEST] auto_threshold={auto_thresh:.4f}')

    # 保存测试结果
    os.makedirs(PIC_DIR, exist_ok=True)
    test_pic = os.path.join(PIC_DIR, 'test_prediction.png')
    pred_mask = (pred >= auto_thresh).astype(np.uint8) * 255

    with _plt_lock:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title('Input')
        axes[0].axis('off')
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('Prediction (thresh=0.3)')
        axes[1].axis('off')
        if os.path.exists(test_mask_path):
            gt = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (IM_WIDTH, IM_HEIGHT))
            axes[2].imshow(gt, cmap='gray')
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
        plt.savefig(test_pic, dpi=72, bbox_inches='tight')
        plt.close()

    return jsonify({
        'message': 'Test prediction completed',
        'image': test_img_path,
        'pred_min': float(pred.min()),
        'pred_max': float(pred.max()),
        'pred_mean': float(pred.mean()),
        'pixels_above_03': int((pred >= 0.3).sum()),
        'result': f'static/pic/test_prediction.png'
    }), 200


@app.route('/eval', methods=['GET'])
def evaluate_model():
    """在测试集上评估模型，返回平均 IoU / Dice。
    可选参数: threshold (默认 0.3), limit (默认 50)"""
    from glob import glob

    thresh = float(request.args.get('threshold', THRESHOLD))
    limit = int(request.args.get('limit', 50))

    test_imgs = glob('D:/Code/code_project/python/project/houai/dataset/srcs/*')
    if not test_imgs:
        return jsonify({'message': '数据集未找到'}), 404

    m = get_model(DEFAULT_MODEL)

    iou_list, dice_list = [], []
    for img_path in test_imgs[:limit]:
        mask_path = img_path.replace('srcs', 'labels_png').replace('.jpg', '.png')
        if not os.path.exists(mask_path):
            continue

        # 预处理（与 unet_pytorch.py adjust_data 完全一致）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT))
        img = img / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (IM_WIDTH, IM_HEIGHT))
        mask_bin = (mask > 127).astype(np.uint8)

        with torch.no_grad():
            pred = m(tensor).cpu().squeeze().numpy()
        pred_bin = (pred >= thresh).astype(np.uint8)

        intersection = (pred_bin & mask_bin).sum()
        union = (pred_bin | mask_bin).sum()
        iou_val = intersection / union if union > 0 else 0.0
        dice_val = 2 * intersection / (pred_bin.sum() + mask_bin.sum()) if (pred_bin.sum() + mask_bin.sum()) > 0 else 0.0

        iou_list.append(iou_val)
        dice_list.append(dice_val)

    return jsonify({
        'num_images': len(iou_list),
        'threshold': thresh,
        'avg_iou': round(float(np.mean(iou_list)), 4),
        'avg_dice': round(float(np.mean(dice_list)), 4),
        'median_iou': round(float(np.median(iou_list)), 4),
        'median_dice': round(float(np.median(dice_list)), 4),
        'iou_list': [round(float(v), 4) for v in iou_list[:10]] + ['...'],
    }), 200


if __name__ == '__main__':
    print('=' * 50)
    print('PyTorch U-Net 喉癌病灶分割 API 启动中 (unet_pytorch)...')
    print(f'Model dir: {MODEL_DIR}')
    print(f'Device: {DEVICE}')
    print('=' * 50)
    # debug=True 会启动 reloader 导致 PyTorch 模型重复加载，用 use_reloader=False
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
