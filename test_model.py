#!/usr/bin/env python3
"""
测试模型加载和结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

def load_model_safely():
    """安全加载模型"""
    model_path = "best_fast_high_accuracy_model.pth"

    if not Path(model_path).exists():
        print(f"模型文件不存在: {model_path}")
        return None

    try:
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"模型加载成功，键: {list(checkpoint.keys())}")

        # 检查模型状态
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        print(f"状态字典包含 {len(state_dict)} 个参数")

        # 显示一些参数信息
        for key, value in list(state_dict.items())[:20]:
            print(f"  {key}: {value.shape}")

        # 特别检查类型分类器的结构
        print("\n类型分类器相关参数:")
        for key, value in state_dict.items():
            if 'type_classifier' in key:
                print(f"  {key}: {value.shape}")

        return state_dict

    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def test_original_model():
    """测试原始模型结构"""
    print("测试原始模型结构...")

    class UltimatePlateModel(nn.Module):
        def __init__(self, num_chars=74, max_length=8, num_plate_types=9):
            super().__init__()
            self.num_chars = num_chars
            self.max_length = max_length
            self.num_plate_types = num_plate_types

            # 使用ResNet34作为骨干网络
            import torchvision.models as models
            resnet = models.resnet34(pretrained=False)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

            # 多级特征提取
            self.feature_pyramid = nn.ModuleList([
                nn.Conv2d(512, 256, 1),
                nn.Conv2d(256, 128, 1),
                nn.Conv2d(128, 64, 1)
            ])

            # 高级注意力机制
            self.attention = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.Sigmoid()
            )

            # 双向GRU序列建模
            self.char_gru = nn.GRU(64, 128, bidirectional=True, batch_first=True, dropout=0.2)

            # 字符分类器
            self.char_classifier = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_chars)
            )

            # 类型分类器
            self.type_classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_plate_types)
            )

            # 位置编码
            self.positional_encoding = nn.Parameter(torch.randn(1, max_length, 256))

        def forward(self, x):
            batch_size = x.size(0)

            # 骨干网络特征提取
            features = self.backbone(x)  # [B, 512, H, W]

            # 特征金字塔
            pyramid_features = []
            for i, conv in enumerate(self.feature_pyramid):
                features = conv(features)
                pyramid_features.append(features)

            # 使用最细粒度的特征
            fine_features = pyramid_features[-1]

            # 注意力机制
            attention_weights = self.attention(fine_features)
            attended_features = fine_features * attention_weights

            # 全局平均池化用于类型分类
            global_features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

            # 序列特征用于字符分类
            seq_features = F.adaptive_avg_pool2d(attended_features, (self.max_length, 1))
            seq_features = seq_features.squeeze(-1).transpose(1, 2)  # [B, L, C]

            # GRU序列建模
            gru_out, _ = self.char_gru(seq_features)

            # 添加位置编码
            gru_out = gru_out + self.positional_encoding

            # 分类
            char_logits = self.char_classifier(gru_out)
            type_logits = self.type_classifier(global_features)

            return char_logits, type_logits

    # 创建模型
    model = UltimatePlateModel()
    print("模型创建成功")

    # 测试前向传播
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)

    print(f"模型输出形状: {output[0].shape}, {output[1].shape}")
    return model

def main():
    print("车牌识别模型测试")
    print("=" * 50)

    # 加载模型状态
    state_dict = load_model_safely()
    if state_dict is None:
        return

    # 测试原始模型结构
    model = test_original_model()

    # 尝试加载权重
    if model:
        try:
            model.load_state_dict(state_dict)
            print("模型权重加载成功！")
        except Exception as e:
            print(f"模型权重加载失败: {e}")

if __name__ == "__main__":
    main()