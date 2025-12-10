#!/usr/bin/env python3
"""
从头开始重新训练系统
完整重新训练所有车牌数据并保存到plans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrainModel(nn.Module):
    """重新训练模型"""
    def __init__(self, num_chars=74, max_length=8, num_plate_types=9):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_plate_types = num_plate_types

        # 使用MobileNetV2作为骨干网络
        mobilenet = models.mobilenet_v2(pretrained=False)
        self.backbone = nn.Sequential(*list(mobilenet.features.children()))

        # 特征增强
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(1280, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1),
            nn.Sigmoid()
        )

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
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_plate_types)
        )

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, 256))

    def forward(self, x):
        batch_size = x.size(0)

        # 特征提取
        features = self.backbone(x)

        # 特征增强
        enhanced_features = self.feature_enhancement(features)

        # 注意力机制
        attention_weights = self.attention(enhanced_features)
        attended_features = enhanced_features * attention_weights

        # 全局平均池化用于类型分类
        global_features = F.adaptive_avg_pool2d(attended_features, (1, 1)).squeeze(-1).squeeze(-1)

        # 序列特征用于字符分类
        seq_features = F.adaptive_avg_pool2d(attended_features, (self.max_length, 1))
        seq_features = seq_features.squeeze(-1).transpose(1, 2)  # [B, L, C]

        # 添加位置编码
        seq_features = seq_features + self.positional_encoding

        # 分类
        char_logits = self.char_classifier(seq_features)
        type_logits = self.type_classifier(global_features)

        return char_logits, type_logits

class RetrainDataset(Dataset):
    """重新训练数据集"""
    def __init__(self, data_dir, label_file, max_samples=None):
        self.data_dir = Path(data_dir)
        self.max_length = 8
        self.max_samples = max_samples

        # 字符集
        self.chars = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领警学挂港澳'
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # 车牌类型
        self.plate_types = [
            '普通蓝牌', '新能源小型车', '新能源大型车', '单层黄牌',
            '黑色车牌', '白色车牌', '双层黄牌', '拖拉机绿牌', '其他类型'
        ]
        self.type_to_idx = {t: idx for idx, t in enumerate(self.plate_types)}
        self.idx_to_type = {idx: t for t, idx in self.type_to_idx.items()}

        # 加载样本
        self.samples = []
        self._load_samples(label_file)

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"重新训练数据集大小: {len(self.samples)}")

    def _load_samples(self, label_file):
        """加载样本数据"""
        with open(label_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if self.max_samples and line_num >= self.max_samples:
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    image_path = parts[0]
                    plate_number = parts[1]
                    plate_type = parts[2]

                    # 检查图像文件是否存在
                    full_path = self.data_dir / image_path
                    if full_path.exists():
                        self.samples.append({
                            'image_path': image_path,
                            'plate_number': plate_number,
                            'plate_type': plate_type
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image_path = self.data_dir / sample['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 编码车牌号码
        plate_number = sample['plate_number']
        encoded_number = []
        for char in plate_number:
            encoded_number.append(self.char_to_idx.get(char, 0))

        # 填充到固定长度
        while len(encoded_number) < self.max_length:
            encoded_number.append(0)
        encoded_number = encoded_number[:self.max_length]

        # 编码车牌类型
        plate_type = sample['plate_type']
        type_idx = self.type_to_idx.get(plate_type, 0)

        return {
            'image': image,
            'plate_number': torch.tensor(encoded_number, dtype=torch.long),
            'plate_type': torch.tensor(type_idx, dtype=torch.long),
            'original_plate_number': plate_number,
            'original_plate_type': plate_type,
            'image_path': str(sample['image_path'])
        }

class RetrainTrainer:
    """重新训练训练器"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 创建数据集
        logger.info("加载训练集...")
        self.train_dataset = RetrainDataset(
            self.data_dir,
            self.data_dir / 'train.txt',
            max_samples=None  # 使用全部训练数据
        )

        logger.info("加载验证集...")
        self.val_dataset = RetrainDataset(
            self.data_dir,
            self.data_dir / 'val.txt',
            max_samples=None  # 使用全部验证数据
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )

        # 创建模型
        self.model = RetrainModel(
            num_chars=len(self.train_dataset.chars),
            max_length=8,
            num_plate_types=len(self.train_dataset.plate_types)
        ).to(self.device)

        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.char_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.type_criterion = nn.CrossEntropyLoss()

        # 初始化权重
        self._initialize_weights()

        logger.info(f"重新训练模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"训练集大小: {len(self.train_dataset):,}")
        logger.info(f"验证集大小: {len(self.val_dataset):,}")

        # 训练历史
        self.train_history = []

    def _initialize_weights(self):
        """初始化权重"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        char_loss_total = 0
        type_loss_total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            plate_numbers = batch['plate_number'].to(self.device)
            plate_types = batch['plate_type'].to(self.device)

            # 前向传播
            char_logits, type_logits = self.model(images)

            # 计算损失
            char_loss = self.char_criterion(
                char_logits.view(-1, char_logits.size(-1)),
                plate_numbers.view(-1)
            )
            type_loss = self.type_criterion(type_logits, plate_types)

            total_batch_loss = char_loss + 0.5 * type_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            char_loss_total += char_loss.item()
            type_loss_total += type_loss.item()

            if batch_idx % 100 == 0:
                logger.info(f'训练 Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                           f'损失: {total_batch_loss.item():.4f} '
                           f'字符损失: {char_loss.item():.4f} '
                           f'类型损失: {type_loss.item():.4f}')

        avg_loss = total_loss / len(self.train_loader)
        avg_char_loss = char_loss_total / len(self.train_loader)
        avg_type_loss = type_loss_total / len(self.train_loader)

        logger.info(f'训练 Epoch {epoch} 完成 - 平均损失: {avg_loss:.4f} '
                   f'字符损失: {avg_char_loss:.4f} 类型损失: {avg_type_loss:.4f}')

        return avg_loss, avg_char_loss, avg_type_loss

    def validate(self):
        """验证模型"""
        self.model.eval()
        vehicle_info = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch['image'].to(self.device)
                plate_numbers = batch['plate_number'].to(self.device)
                plate_types = batch['plate_type'].to(self.device)

                # 前向传播
                char_logits, type_logits = self.model(images)

                # 获取预测结果
                char_preds = char_logits.argmax(dim=-1)
                type_preds = type_logits.argmax(dim=-1)

                # 处理每个样本
                for i in range(len(batch['image_path'])):
                    image_path = batch['image_path'][i]
                    true_plate_number = batch['original_plate_number'][i]
                    true_plate_type = batch['original_plate_type'][i]

                    # 解码预测结果
                    pred_chars = []
                    for j in range(self.val_dataset.max_length):
                        char_idx = char_preds[i, j].item()
                        if char_idx > 0:  # 不是padding
                            pred_chars.append(self.val_dataset.idx_to_char.get(char_idx, ''))
                        else:
                            break
                    pred_plate_number = ''.join(pred_chars)
                    pred_plate_type = self.val_dataset.idx_to_type.get(type_preds[i].item(), '其他类型')

                    vehicle_info.append({
                        'image_path': image_path,
                        'true_plate_number': true_plate_number,
                        'true_plate_type': true_plate_type,
                        'pred_plate_number': pred_plate_number,
                        'pred_plate_type': pred_plate_type,
                        'is_correct_number': pred_plate_number == true_plate_number,
                        'is_correct_type': pred_plate_type == true_plate_type
                    })

                if batch_idx % 20 == 0:
                    logger.info(f'验证进度: {batch_idx}/{len(self.val_loader)}')

        # 计算准确率
        total_samples = len(vehicle_info)
        correct_numbers = sum(1 for v in vehicle_info if v['is_correct_number'])
        correct_types = sum(1 for v in vehicle_info if v['is_correct_type'])

        char_accuracy = correct_numbers / total_samples
        type_accuracy = correct_types / total_samples
        overall_accuracy = (char_accuracy + type_accuracy) / 2

        logger.info(f"验证完成!")
        logger.info(f"  车牌号码准确率: {char_accuracy:.6f} ({correct_numbers}/{total_samples})")
        logger.info(f"  车牌类型准确率: {type_accuracy:.6f} ({correct_types}/{total_samples})")
        logger.info(f"  综合准确率: {overall_accuracy:.6f}")

        return vehicle_info, char_accuracy, type_accuracy, overall_accuracy

    def train(self, num_epochs=10):
        """完整训练流程"""
        logger.info("开始从头训练...")

        best_accuracy = 0
        best_epoch = 0
        best_results = None

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # 训练
            train_loss, char_loss, type_loss = self.train_epoch(epoch)

            # 验证
            vehicle_info, char_acc, type_acc, overall_acc = self.validate()

            # 保存训练历史
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'char_loss': char_loss,
                'type_loss': type_loss,
                'char_accuracy': char_acc,
                'type_accuracy': type_acc,
                'overall_accuracy': overall_acc
            })

            # 保存最佳结果
            if overall_acc > best_accuracy:
                best_accuracy = overall_acc
                best_epoch = epoch + 1
                best_results = (vehicle_info, char_acc, type_acc, overall_acc)

            logger.info(f"当前最佳准确率: {best_accuracy:.6f} (Epoch {best_epoch})")

        logger.info(f"训练完成! 最佳准确率: {best_accuracy:.6f} (Epoch {best_epoch})")
        return best_results, self.train_history

    def save_training_results(self, best_results, train_history):
        """保存训练结果到plans"""
        vehicle_info, char_acc, type_acc, overall_acc = best_results

        plans_dir = Path("C:/Users/ASUS/Desktop/科研+论文/车牌识别/plans")
        plans_dir.mkdir(exist_ok=True)

        with open(plans_dir / "plans.txt", 'w', encoding='utf-8') as f:
            f.write("重新训练车牌识别系统结果报告\n")
            f.write("=" * 120 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集路径: {self.data_dir}\n")
            f.write(f"训练集大小: {len(self.train_dataset):,}\n")
            f.write(f"验证集大小: {len(self.val_dataset):,}\n")
            f.write(f"总数据量: {len(self.train_dataset) + len(self.val_dataset):,}\n")
            f.write(f"模型类型: RetrainModel (MobileNetV2 + Feature Enhancement + Attention)\n")
            f.write(f"训练策略: 从头开始训练\n")
            f.write("=" * 120 + "\n\n")

            # 训练历史
            f.write("训练历史:\n")
            f.write("-" * 120 + "\n")
            for history in train_history:
                f.write(f"Epoch {history['epoch']:2d}: "
                       f"损失={history['train_loss']:.4f}, "
                       f"字符准确率={history['char_accuracy']:.4f}, "
                       f"类型准确率={history['type_accuracy']:.4f}, "
                       f"综合准确率={history['overall_accuracy']:.4f}\n")
            f.write("=" * 120 + "\n\n")

            # 最终统计信息
            total_samples = len(vehicle_info)
            correct_numbers = sum(1 for v in vehicle_info if v['is_correct_number'])
            correct_types = sum(1 for v in vehicle_info if v['is_correct_type'])

            f.write("最终训练结果:\n")
            f.write(f"  总验证样本数: {total_samples:,}\n")
            f.write(f"  车牌号码正确数: {correct_numbers:,}\n")
            f.write(f"  车牌号码准确率: {correct_numbers/total_samples:.6f}\n")
            f.write(f"  车牌类型正确数: {correct_types:,}\n")
            f.write(f"  车牌类型准确率: {correct_types/total_samples:.6f}\n")
            f.write(f"  综合准确率: {(correct_numbers + correct_types) / (2 * total_samples):.6f}\n")
            f.write(f"  错误样本数: {total_samples - correct_numbers}\n")
            f.write(f"  错误率: {(total_samples - correct_numbers) / total_samples:.6f}\n")
            f.write("=" * 120 + "\n\n")

            # 详细车辆信息 (前500个)
            f.write("详细车辆信息 (前500个样本):\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'序号':<8} {'图片路径':<50} {'真实车牌':<12} {'预测车牌':<12} {'真实类型':<12} {'预测类型':<12} {'结果':<8}\n")
            f.write("-" * 120 + "\n")

            for i, vehicle in enumerate(vehicle_info[:500]):
                result_status = "✓" if vehicle['is_correct_number'] and vehicle['is_correct_type'] else "✗"
                f.write(f"{i+1:<8} {vehicle['image_path']:<50} "
                       f"{vehicle['true_plate_number']:<12} {vehicle['pred_plate_number']:<12} "
                       f"{vehicle['true_plate_type']:<12} {vehicle['pred_plate_type']:<12} "
                       f"{result_status:<8}\n")

            # 错误样本分析
            error_samples = [v for v in vehicle_info if not (v['is_correct_number'] and v['is_correct_type'])]
            f.write(f"\n错误样本分析 (共{len(error_samples)}个):\n")
            f.write("-" * 120 + "\n")
            for i, error in enumerate(error_samples[:50]):  # 只显示前50个错误
                error_type = []
                if not error['is_correct_number']:
                    error_type.append("号码错误")
                if not error['is_correct_type']:
                    error_type.append("类型错误")
                f.write(f"{i+1:<4} {error['image_path']:<50} "
                       f"{error['true_plate_number']:<12} {error['pred_plate_number']:<12} "
                       f"{error['true_plate_type']:<12} {error['pred_plate_type']:<12} "
                       f"{','.join(error_type):<8}\n")

            # 车牌类型分布
            type_distribution = {}
            for vehicle in vehicle_info:
                true_type = vehicle['true_plate_type']
                type_distribution[true_type] = type_distribution.get(true_type, 0) + 1

            f.write("\n车牌类型分布:\n")
            for plate_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_samples * 100
                f.write(f"  {plate_type}: {count:,} ({percentage:.2f}%)\n")

            # 技术分析
            f.write("\n" + "=" * 120 + "\n")
            f.write("技术分析:\n")
            f.write("-" * 120 + "\n")
            f.write(f"  模型参数量: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"  训练集规模: {len(self.train_dataset):,} 样本\n")
            f.write(f"  验证集规模: {len(self.val_dataset):,} 样本\n")
            f.write(f"  总数据规模: {len(self.train_dataset) + len(self.val_dataset):,} 样本\n")
            f.write(f"  模型架构: MobileNetV2 + Feature Enhancement + Attention\n")
            f.write(f"  优化器: Adam\n")
            f.write(f"  学习率: 1e-3\n")
            f.write(f"  训练轮数: {len(train_history)}\n")
            f.write(f"  性能评级: {'优秀' if overall_acc > 0.95 else '良好' if overall_acc > 0.9 else '一般'}\n")

        logger.info(f"训练结果已保存到: {plans_dir / 'plans.txt'}")

def main():
    """主函数"""
    # 配置路径
    data_dir = "C:/Users/ASUS/Desktop/科研+论文/车牌识别/CBLPRD-330k_v1"

    # 创建训练器
    trainer = RetrainTrainer(data_dir)

    # 开始训练
    best_results, train_history = trainer.train(num_epochs=5)

    # 保存训练结果
    trainer.save_training_results(best_results, train_history)

    vehicle_info, char_acc, type_acc, overall_acc = best_results

    logger.info("重新训练完成！")
    logger.info(f"最终综合准确率: {overall_acc:.6f}")
    logger.info(f"车牌号码准确率: {char_acc:.6f}")
    logger.info(f"车牌类型准确率: {type_acc:.6f}")
    logger.info("训练结果已保存到plans文件")

if __name__ == "__main__":
    main()