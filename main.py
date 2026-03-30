import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
from data_load import PreprocessedEEGDataset
from model import EnhancedDualDomainJointModel,EnhancedLoss
from train import train_enhanced_model

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n=== Device Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {device}")
print(f"Device name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")

# ==================== 训练参数 ====================
WINDOW_SIZE = 128
OVERLAP_RATIO = 0
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 40
LR = 0.001
TRAIN_VAL_SPLIT = 0.8

# 数据保存路径
DATA_DIR = "./"

# ==================== 主函数 ====================

def main():
    """主函数 - 直接加载已生成的.mat文件进行训练"""

    snr = -3

    batch_size = BATCH_SIZE
    max_epochs = EPOCHS
    lr = LR

    results = []

    print("\n" + "=" * 80)
    print("Loading Preprocessed Data and Training Models")
    print("=" * 80)
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")

    # 检查数据目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"\nError: Data directory '{DATA_DIR}' does not exist!")
        print("Please run data preprocessing first to generate the .mat files.")
        return

    # 构建数据文件路径
    data_file = os.path.join(DATA_DIR, f"eeg_data_snr_{snr}.mat")


    print(f"\n{'=' * 80}")
    print(f"Training Enhanced Dual-Domain Mamba (SNR={snr}dB)")
    print(f"{'=' * 80}")

    # 从保存的.mat文件加载数据集
    dataset = PreprocessedEEGDataset(data_file)

    # 划分数据集
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0, drop_last=True
    )

    # 初始化模型
    model = EnhancedDualDomainJointModel(
        in_channels=32,
        length=128,
        num_classes=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    criterion = EnhancedLoss(
        denoise_weight=3.0,
        classify_weight=1.0,
        num_classes=4
    ).to(device)

    # 训练
    best_metrics = train_enhanced_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        max_epochs=max_epochs,
        patience=PATIENCE,
        snr=snr
    )

    # 记录结果
    results.append({
        'SNR': snr,
        'Val_Accuracy': best_metrics['acc'],
        'Val_Macro_F1': best_metrics['f1'],
        'Denoise_SNR': best_metrics['snr'],
        'Correlation_Coeff': best_metrics['cc'],
        'Denoise_MSE': best_metrics['mse'],
        'Model': 'Preprocessed_Data_DualMamba'
    })

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv("preprocessed_data_training_results.csv", index=False)
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(df.to_string())
        print("\nResults saved to: preprocessed_data_training_results.csv")
    else:
        print("\nNo models were trained. Please check data files.")


if __name__ == "__main__":
    main()