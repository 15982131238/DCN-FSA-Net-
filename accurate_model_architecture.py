#!/usr/bin/env python3
"""
基于trained weights的准确模型架构重建
从best_fast_high_accuracy_model.pth分析得出的准确架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActualTrainedModel(nn.Module):
    """基于trained weights的准确模型架构重建"""

    def __init__(self, num_chars=72, max_length=8, num_plate_types=9):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_plate_types = num_plate_types

        # === BACKBONE结构 (基于trained weights分析) ===
        # Layer 0: 初始卷积层
        self.backbone_0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # Layer 1: BatchNorm2d
        self.backbone_1 = nn.BatchNorm2d(64)

        # Layer 4: 第一个残差块 (64->64)
        self.backbone_4_0_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_0_bn1 = nn.BatchNorm2d(64)
        self.backbone_4_0_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_0_bn2 = nn.BatchNorm2d(64)

        self.backbone_4_1_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_1_bn1 = nn.BatchNorm2d(64)
        self.backbone_4_1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_1_bn2 = nn.BatchNorm2d(64)

        # Layer 5: 第二个残差块 (64->128, with downsampling)
        self.backbone_5_0_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.backbone_5_0_bn1 = nn.BatchNorm2d(128)
        self.backbone_5_0_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.backbone_5_0_bn2 = nn.BatchNorm2d(128)
        self.backbone_5_0_downsample_0 = nn.Conv2d(64, 128, kernel_size=1)
        self.backbone_5_0_downsample_1 = nn.BatchNorm2d(128)

        self.backbone_5_1_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.backbone_5_1_bn1 = nn.BatchNorm2d(128)
        self.backbone_5_1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.backbone_5_1_bn2 = nn.BatchNorm2d(128)

        # Layer 6: 第三个残差块 (128->256, with downsampling)
        self.backbone_6_0_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.backbone_6_0_bn1 = nn.BatchNorm2d(256)
        self.backbone_6_0_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.backbone_6_0_bn2 = nn.BatchNorm2d(256)
        self.backbone_6_0_downsample_0 = nn.Conv2d(128, 256, kernel_size=1)
        self.backbone_6_0_downsample_1 = nn.BatchNorm2d(256)

        self.backbone_6_1_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.backbone_6_1_bn1 = nn.BatchNorm2d(256)
        self.backbone_6_1_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.backbone_6_1_bn2 = nn.BatchNorm2d(256)

        # Layer 7: 第四个残差块 (256->512, with downsampling)
        self.backbone_7_0_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.backbone_7_0_bn1 = nn.BatchNorm2d(512)
        self.backbone_7_0_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.backbone_7_0_bn2 = nn.BatchNorm2d(512)
        self.backbone_7_0_downsample_0 = nn.Conv2d(256, 512, kernel_size=1)
        self.backbone_7_0_downsample_1 = nn.BatchNorm2d(512)

        self.backbone_7_1_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.backbone_7_1_bn1 = nn.BatchNorm2d(512)
        self.backbone_7_1_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.backbone_7_1_bn2 = nn.BatchNorm2d(512)

        # === FEATURE ENHANCEMENT结构 ===
        self.feature_enhancement_0 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.feature_enhancement_1 = nn.BatchNorm2d(256)
        self.feature_enhancement_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.feature_enhancement_5 = nn.BatchNorm2d(128)

        # === ATTENTION机制 (使用FC layers) ===
        self.attention_fc_0 = nn.Linear(512, 64)  # 512->64
        self.attention_fc_2 = nn.Linear(64, 512)   # 64->512

        # === 分类器结构 ===
        # 字符分类器: 128->64->72
        self.char_classifier_0 = nn.Linear(128, 64)
        self.char_classifier_3 = nn.Linear(64, 72)  # 注意: trained weights显示是72个字符

        # 类型分类器: 128->64->9
        self.type_classifier_0 = nn.Linear(128, 64)
        self.type_classifier_3 = nn.Linear(64, 9)

        # === 位置编码 ===
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, 128))

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.size(0)

        # === BACKBONE特征提取 ===
        # Layer 0: Conv2d
        x = self.backbone_0(x)
        x = self.relu(x)

        # Layer 1: BatchNorm2d
        x = self.backbone_1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Layer 4: 第一个残差块
        identity = x
        out = self.relu(self.backbone_4_0_bn1(self.backbone_4_0_conv1(x)))
        out = self.backbone_4_0_bn2(self.backbone_4_0_conv2(out))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_4_1_bn1(self.backbone_4_1_conv1(x)))
        out = self.backbone_4_1_bn2(self.backbone_4_1_conv2(out))
        out += identity
        x = self.relu(out)

        # Layer 5: 第二个残差块 (64->128)
        identity = x
        out = self.relu(self.backbone_5_0_bn1(self.backbone_5_0_conv1(x)))
        out = self.backbone_5_0_bn2(self.backbone_5_0_conv2(out))
        identity = self.backbone_5_0_downsample_1(self.backbone_5_0_downsample_0(identity))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_5_1_bn1(self.backbone_5_1_conv1(x)))
        out = self.backbone_5_1_bn2(self.backbone_5_1_conv2(out))
        out += identity
        x = self.relu(out)

        # Layer 6: 第三个残差块 (128->256)
        identity = x
        out = self.relu(self.backbone_6_0_bn1(self.backbone_6_0_conv1(x)))
        out = self.backbone_6_0_bn2(self.backbone_6_0_conv2(out))
        identity = self.backbone_6_0_downsample_1(self.backbone_6_0_downsample_0(identity))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_6_1_bn1(self.backbone_6_1_conv1(x)))
        out = self.backbone_6_1_bn2(self.backbone_6_1_conv2(out))
        out += identity
        x = self.relu(out)

        # Layer 7: 第四个残差块 (256->512)
        identity = x
        out = self.relu(self.backbone_7_0_bn1(self.backbone_7_0_conv1(x)))
        out = self.backbone_7_0_bn2(self.backbone_7_0_conv2(out))
        identity = self.backbone_7_0_downsample_1(self.backbone_7_0_downsample_0(identity))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_7_1_bn1(self.backbone_7_1_conv1(x)))
        out = self.backbone_7_1_bn2(self.backbone_7_1_conv2(out))
        out += identity
        x = self.relu(out)

        # 保存512维特征用于注意力
        features_512 = x  # [B, 512, H, W]

        # === FEATURE ENHANCEMENT ===
        x = self.relu(self.feature_enhancement_1(self.feature_enhancement_0(x)))
        x = self.relu(self.feature_enhancement_5(self.feature_enhancement_4(x)))
        features_128 = x  # [B, 128, H, W]

        # === ATTENTION机制 ===
        # Global average pooling to get 512 features
        global_features = F.adaptive_avg_pool2d(features_512, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 512]

        # Apply attention using FC layers
        attention_weights = self.attention_fc_0(global_features)  # [B, 64]
        attention_weights = torch.sigmoid(attention_weights)  # [B, 64]
        attention_weights = self.attention_fc_2(attention_weights)  # [B, 512]
        attention_weights = torch.sigmoid(attention_weights)  # [B, 512]

        # Reshape attention weights to match spatial dimensions
        # 这里需要根据实际的feature map大小来调整
        B, C, H, W = features_512.shape
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1]
        attention_weights = attention_weights.expand(-1, -1, H, W)  # [B, 512, H, W]

        # Apply attention to 512 features
        attended_features = features_512 * attention_weights

        # 进一步降维到128维
        attended_features = self.relu(self.feature_enhancement_1(self.feature_enhancement_0(attended_features)))
        attended_features = self.relu(self.feature_enhancement_5(self.feature_enhancement_4(attended_features)))

        # === 全局平均池化用于类型分类 ===
        global_features = F.adaptive_avg_pool2d(attended_features, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 128]

        # === 序列特征用于字符分类 ===
        seq_features = F.adaptive_avg_pool2d(features_128, (self.max_length, 1))  # [B, 128, L, 1]
        seq_features = seq_features.squeeze(-1).transpose(1, 2)  # [B, L, 128]

        # === 添加位置编码 ===
        seq_features = seq_features + self.positional_encoding

        # === 分类 ===
        char_logits = self.char_classifier_3(self.relu(self.char_classifier_0(seq_features)))
        type_logits = self.type_classifier_3(self.relu(self.type_classifier_0(global_features)))

        return char_logits, type_logits

def load_trained_model(model_path="best_fast_high_accuracy_model.pth"):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = ActualTrainedModel(num_chars=72, max_length=8, num_plate_types=9)
    model.to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)

    # 重命名权重名称以匹配新模型结构
    new_state_dict = {}
    for key, value in checkpoint.items():
        # 将点分隔的层名映射到新的下划线分隔的层名
        new_key = key.replace('.', '_')
        new_state_dict[new_key] = value

    # 尝试加载权重
    try:
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("模型权重加载成功!")
    except Exception as e:
        logger.error(f"模型权重加载失败: {e}")
        # 尝试部分加载
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in new_state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                logger.warning(f"跳过参数: {k}, 形状不匹配: {v.shape} vs {model_dict.get(k, torch.Tensor()).shape}")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"成功加载 {len(pretrained_dict)}/{len(new_state_dict)} 个参数")

    model.eval()
    return model

def test_model():
    """测试模型"""
    print("测试基于trained weights的准确模型架构...")

    # 加载模型
    model = load_trained_model()

    # 测试前向传播
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        char_logits, type_logits = model(test_input)

    print(f"字符分类器输出形状: {char_logits.shape}")
    print(f"类型分类器输出形状: {type_logits.shape}")
    print("模型测试成功!")

if __name__ == "__main__":
    test_model()