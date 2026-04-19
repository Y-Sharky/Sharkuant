# cnn_predictor.py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ==================== 图像生成器（与训练时一致）====================
def generate_stock_image(df, image_size=(112, 112), ma_window=20):
    """
    生成单窗口K线图像（灰度图）
    df: 包含 open, high, low, close, vol 的DataFrame
    image_size: (width, height)
    """
    if len(df) == 0:
        return None
    plot_df = df.copy()
    max_price = plot_df[['high', 'close', 'open']].max().max()
    min_price = plot_df[['low', 'close', 'open']].min().min()
    price_range = max_price - min_price
    if price_range == 0:
        price_range = 1.0

    def price_to_y(p):
        return int((1 - (p - min_price) / price_range) * 0.8 * image_size[1]) + int(0.1 * image_size[1])

    max_vol = plot_df['vol'].max()
    vol_height = int(image_size[1] * 0.2)
    img = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(img)
    n_days = len(plot_df)
    bar_width = image_size[0] / n_days

    for i, (_, row) in enumerate(plot_df.iterrows()):
        x_center = int(i * bar_width + bar_width/2)
        x_left = int(i * bar_width)
        x_right = int((i+1) * bar_width)
        y_high = price_to_y(row['high'])
        y_low = price_to_y(row['low'])
        draw.line([(x_center, y_high), (x_center, y_low)], fill=255, width=1)
        y_open = price_to_y(row['open'])
        y_close = price_to_y(row['close'])
        rect_width = int(bar_width * 0.6)
        draw.rectangle([x_center - rect_width//2, min(y_open, y_close),
                        x_center + rect_width//2, max(y_open, y_close)],
                       fill=255, outline=255)
        vol_bar_height = int((row['vol'] / max_vol) * vol_height) if max_vol > 0 else 0
        draw.rectangle([x_left, image_size[1] - vol_bar_height,
                        x_right, image_size[1]], fill=255)

    if len(plot_df) >= 5:
        ma5 = plot_df['close'].rolling(5).mean()
        points = []
        for i, (date, _) in enumerate(plot_df.iterrows()):
            if pd.notna(ma5.loc[date]):
                x = int(i * bar_width + bar_width/2)
                y = price_to_y(ma5.loc[date])
                points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=128, width=1)
    return np.array(img) / 255.0   # 归一化到 [0,1]

# ==================== 多尺度 CNN 模型（与训练脚本完全一致）====================
class MultiChannelStockCNN(nn.Module):
    """带BatchNorm的多尺度CNN，输入 (3,112,112)"""
    def __init__(self):
        super(MultiChannelStockCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== 加载模型（支持单模型和多模型集成）====================
def load_single_model(model_path: str, device: torch.device = None) -> nn.Module:
    """加载单个 MultiChannelStockCNN 模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiChannelStockCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_ensemble_models(model_paths: List[str], device: torch.device = None) -> List[nn.Module]:
    """加载多个模型用于集成"""
    models = []
    for path in model_paths:
        if not path:
            continue
        try:
            model = load_single_model(path, device)
            models.append(model)
            print(f"Loaded model: {path}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return models

def predict_with_cnn(model, device, image_array):
    """
    单张图像预测，返回上涨概率
    image_array: (3, H, W) numpy array, 值域 [0,1]
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be numpy array")
    img_tensor = torch.FloatTensor(image_array).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        prob = F.softmax(outputs, dim=1)[0, 1].item()
    return prob

def predict_batch_with_cnn(model, device, image_arrays):
    """
    批量预测，返回上涨概率列表
    image_arrays: list of (3, H, W) numpy arrays
    """
    if not image_arrays:
        return []
    batch_tensor = torch.FloatTensor(np.stack(image_arrays)).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().tolist()
    return probs

def predict_ensemble(models, device, image_array):
    """集成预测：多个模型平均概率"""
    probs = []
    for model in models:
        prob = predict_with_cnn(model, device, image_array)
        probs.append(prob)
    return np.mean(probs)

# 如果直接运行，测试模型加载
if __name__ == "__main__":
    # 测试图像生成
    from datetime import datetime
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    df = pd.DataFrame({
        'open': np.random.rand(60) * 100 + 100,
        'high': np.random.rand(60) * 110 + 100,
        'low': np.random.rand(60) * 90 + 100,
        'close': np.random.rand(60) * 100 + 100,
        'vol': np.random.rand(60) * 1e7
    }, index=dates)
    img = generate_stock_image(df, image_size=(112,112))
    print("图像生成测试:", img.shape if img is not None else "失败")