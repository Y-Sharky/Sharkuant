"""
train_news_model_simple.py (回归版本)
- 影响力任务改为回归 (HuberLoss)
- 恢复训练时仅加载模型权重（优化器/调度器重新初始化）
- 数据增强使用 nlpaug 同义词替换（可选）
- 支持正文拼接、可学习损失权重
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel

try:
    from transformers import AdamW
except ImportError:
    try:
        from transformers.optimization import AdamW
    except ImportError:
        from torch.optim import AdamW

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from train_news_model_utils_simple import MultiTaskFinBERT, NewsDataset

# 可选数据增强（使用同义词替换）
try:
    import nlpaug.augmenter.word as naw
    AUG_AVAILABLE = True
except ImportError:
    AUG_AVAILABLE = False
    print("nlpaug 未安装，将跳过数据增强。安装命令: pip install nlpaug")

# GPU 加速设置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)} 可用，启用加速。")
else:
    print("GPU 不可用，使用 CPU。")


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="改进版新闻分析模型训练（回归版）")
    parser.add_argument('--data', type=str, default='training_data/labeled_news.csv', help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='models/news_model_deberta', help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base', help='预训练模型名称')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max_len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--test_size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径（仅模型权重）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='预热步数占总步数的比例')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪的最大范数')
    # 新增参数
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='分类头前的 dropout 比率')
    parser.add_argument('--use_content', action='store_true', help='是否使用正文内容（需数据集中有 content 列）')
    parser.add_argument('--augment', action='store_true', help='是否对训练集进行文本增强（需安装 nlpaug）')
    parser.add_argument('--augment_ratio', type=float, default=0.3, help='增强比例（对训练集中部分样本进行增强）')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma 参数')
    parser.add_argument('--early_stop_metric', type=str, default='f1', choices=['loss', 'f1'], help='早停指标')
    parser.add_argument('--learnable_weights', action='store_true', help='是否使用可学习损失权重')
    # 注意：回归版本不再需要 impact_classes 参数
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = ['title', 'news_type', 'sentiment', 'impact']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    return df


def prepare_labels(df):
    """直接返回原始 impact 值（不离散化）"""
    type_encoder = LabelEncoder()
    type_labels = type_encoder.fit_transform(df['news_type'].fillna('其他'))
    sentiment_labels = df['sentiment'].fillna(0.0).values.astype(np.float32)
    impact_labels = df['impact'].fillna(0).values.astype(np.float32)   # 原始值 0~5
    return type_labels, sentiment_labels, impact_labels, type_encoder


def synonym_augment(texts, labels, aug_ratio=0.3):
    """使用同义词替换进行增强，返回增强后的文本和对应标签（长度增加）"""
    if not AUG_AVAILABLE:
        print("警告: nlpaug 未安装，跳过数据增强。")
        return texts, labels
    # 使用英文词库，中文需指定 wordnet 或自定义词典
    aug = naw.SynonymAug(aug_src='wordnet', lang='eng')
    new_texts = []
    new_labels = []
    for text, label in zip(texts, labels):
        new_texts.append(text)
        new_labels.append(label)
        if random.random() < aug_ratio:
            try:
                augmented = aug.augment(text)
                if augmented and augmented != text:
                    new_texts.append(augmented)
                    new_labels.append(label)
            except Exception as e:
                print(f"增强失败: {e}")
    return new_texts, new_labels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------- 数据加载 --------------------
    df = load_data(args.data)
    print(f"加载数据 {len(df)} 条")

    use_content = args.use_content and 'content' in df.columns
    if use_content:
        print("使用正文内容（标题 + [SEP] + 正文）")
        contents = df['content'].fillna('').tolist()
    else:
        contents = [None] * len(df)

    type_labels, sentiment_labels, impact_labels, type_encoder = prepare_labels(df)
    num_types = len(type_encoder.classes_)
    print(f"新闻类型类别数: {num_types} 类别: {type_encoder.classes_}")

    texts = df['title'].tolist()

    # -------------------- 划分数据集 --------------------
    (train_texts, val_texts,
     train_type, val_type,
     train_sent, val_sent,
     train_imp, val_imp,
     train_contents, val_contents) = train_test_split(
        texts, type_labels, sentiment_labels, impact_labels, contents,
        test_size=args.test_size, random_state=args.seed, stratify=type_labels
    )

    # -------------------- 数据增强 --------------------
    if args.augment and args.augment_ratio > 0:
        print(f"进行文本增强，增强比例 {args.augment_ratio}")
        aug_texts, aug_types = synonym_augment(train_texts, train_type, args.augment_ratio)
        orig_len = len(train_texts)
        train_texts = aug_texts
        train_type = aug_types
        added = len(train_texts) - orig_len
        if added > 0:
            # 简单重复其他标签以匹配增强后的样本数量
            train_sent = np.concatenate([train_sent] * (added // orig_len + 1))[:len(train_texts)]
            train_imp = np.concatenate([train_imp] * (added // orig_len + 1))[:len(train_texts)]
            if use_content:
                train_contents = train_contents * (added // orig_len + 1)
                train_contents = train_contents[:len(train_texts)]
        print(f"增强后训练集大小: {len(train_texts)}")

    # -------------------- 类别权重 --------------------
    class_counts = np.bincount(train_type)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    type_classes = type_encoder.classes_.tolist()
    if '其他' in type_classes:
        idx = type_classes.index('其他')
        class_weights[idx] *= 2.5
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"类型类别权重: {class_weights.cpu().numpy()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # -------------------- 构建 DataLoader --------------------
    train_dataset = NewsDataset(train_texts, train_type, train_sent, train_imp,
                                tokenizer, args.max_len, use_content=use_content,
                                content_list=train_contents if use_content else None)
    val_dataset = NewsDataset(val_texts, val_type, val_sent, val_imp,
                              tokenizer, args.max_len, use_content=use_content,
                              content_list=val_contents if use_content else None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == 'cuda'))

    # -------------------- 模型 --------------------
    base_model = AutoModel.from_pretrained(args.model_name)
    model = MultiTaskFinBERT(base_model, num_types, dropout_rate=args.dropout_rate)   # 回归模型
    model.to(device)
    model = model.float()

    # -------------------- 恢复模型权重（如果指定）--------------------
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        # 只加载模型权重，忽略形状不匹配的层（如回归头与分类头维度不同）
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"恢复模型权重，从 epoch {start_epoch} 开始")
        print("注意：优化器与调度器状态未恢复，已重新初始化。")

    # -------------------- 总步数与预热 --------------------
    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    print(f"总步数: {total_steps}, 预热步数: {num_warmup_steps}")

    # -------------------- 可学习权重参数 --------------------
    log_var_type = log_var_sent = log_var_imp = None
    if args.learnable_weights:
        log_var_type = torch.nn.Parameter(torch.zeros(1, device=device))
        log_var_sent = torch.nn.Parameter(torch.zeros(1, device=device))
        log_var_imp = torch.nn.Parameter(torch.zeros(1, device=device))

    # 优化器与调度器总是全新创建（避免形状不匹配）
    if args.learnable_weights:
        optim_params = list(model.parameters()) + [log_var_type, log_var_sent, log_var_imp]
    else:
        optim_params = model.parameters()

    optimizer = AdamW(optim_params, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    # -------------------- 损失函数 --------------------
    type_loss_fn = FocalLoss(gamma=args.focal_gamma, weight=class_weights).to(device)
    sentiment_loss_fn = torch.nn.HuberLoss(delta=1.0)          # 情感回归损失
    impact_loss_fn = torch.nn.HuberLoss(delta=1.0)             # 影响力回归损失

    # -------------------- 训练循环 --------------------
    best_val_metric = float('inf') if args.early_stop_metric == 'loss' else 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            type_label = batch['type_label'].to(device)
            sentiment_label = batch['sentiment_label'].to(device)
            impact_label = batch['impact_label'].to(device)   # 浮点数

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss_type = type_loss_fn(outputs['type_logits'], type_label)
            loss_sent = sentiment_loss_fn(outputs['sentiment'], sentiment_label)
            loss_imp = impact_loss_fn(outputs['impact'], impact_label)   # 回归损失

            if args.learnable_weights:
                loss = (torch.exp(-log_var_type) * loss_type + log_var_type +
                        torch.exp(-log_var_sent) * loss_sent + log_var_sent +
                        torch.exp(-log_var_imp) * loss_imp + log_var_imp)
            else:
                loss = loss_type + loss_sent + loss_imp

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # -------------------- 验证 --------------------
        model.eval()
        val_loss_total = 0
        all_type_pred, all_type_true = [], []
        all_sent_pred, all_sent_true = [], []
        all_impact_pred, all_impact_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)

                loss_type = type_loss_fn(outputs['type_logits'], batch['type_label'].to(device))
                val_loss_total += loss_type.item()

                preds = torch.argmax(outputs['type_logits'], dim=-1).cpu().numpy()
                all_type_pred.extend(preds)
                all_type_true.extend(batch['type_label'].cpu().numpy())

                all_sent_pred.extend(outputs['sentiment'].cpu().numpy())
                all_sent_true.extend(batch['sentiment_label'].cpu().numpy())

                all_impact_pred.extend(outputs['impact'].cpu().numpy())
                all_impact_true.extend(batch['impact_label'].cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader)
        val_acc = accuracy_score(all_type_true, all_type_pred)
        val_f1 = f1_score(all_type_true, all_type_pred, average='weighted')

        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)

        # 计算 RMSE
        sent_rmse = np.sqrt(np.mean((np.array(all_sent_true) - np.array(all_sent_pred))**2))
        imp_rmse = np.sqrt(np.mean((np.array(all_impact_true) - np.array(all_impact_pred))**2))

        print(f"Epoch {epoch+1} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_f1: {val_f1:.4f}, val_acc: {val_acc:.4f}")
        print(f"  情感 RMSE: {sent_rmse:.4f}, 影响力 RMSE: {imp_rmse:.4f}")

        report = classification_report(all_type_true, all_type_pred,
                                       target_names=type_encoder.classes_,
                                       output_dict=True, zero_division=0)
        for cls in type_encoder.classes_:
            if cls in report:
                print(f"  {cls} F1: {report[cls]['f1-score']:.4f}")

        # 早停判断
        if args.early_stop_metric == 'loss':
            current_metric = avg_val_loss
            is_better = current_metric < best_val_metric
        else:
            current_metric = val_f1
            is_better = current_metric > best_val_metric

        if is_better:
            best_val_metric = current_metric
            patience_counter = 0
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
                'f1': val_f1,
                # 回归模型不保存 impact_classes
            }
            if args.learnable_weights:
                save_dict.update({
                    'log_var_type': log_var_type,
                    'log_var_sent': log_var_sent,
                    'log_var_imp': log_var_imp,
                })
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
            encoders_info = {'type_classes': type_encoder.classes_.tolist()}
            with open(os.path.join(args.output_dir, 'encoders.json'), 'w') as f:
                json.dump(encoders_info, f)
            tokenizer.save_pretrained(args.output_dir)
            print(f"最佳模型已保存 (val_{args.early_stop_metric}={best_val_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停: {args.patience} 个 epoch 未改善，停止训练。")
                break

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, 'history.csv'), index=False)
    print("训练完成！")


if __name__ == "__main__":
    main()