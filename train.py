import sys

sys.path.insert(0, '/media/aaa/d6249A89249A6BED/anaconda3/envs/mamba310/lib/python3.10/site-packages')

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_snr(clean, denoised):
    signal_power = torch.var(clean)
    noise_power = torch.var(clean - denoised)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr.item()


def calculate_cc(clean, denoised):
    clean_flat = clean.contiguous().view(clean.size(0), -1)
    denoised_flat = denoised.contiguous().view(denoised.size(0), -1)
    clean_mean = clean_flat - clean_flat.mean(dim=1, keepdim=True)
    denoised_mean = denoised_flat - denoised_flat.mean(dim=1, keepdim=True)
    covariance = (clean_mean * denoised_mean).sum(dim=1)
    clean_std = torch.sqrt((clean_mean ** 2).sum(dim=1))
    denoised_std = torch.sqrt((denoised_mean ** 2).sum(dim=1))
    cc = covariance / (clean_std * denoised_std + 1e-8)
    return cc.mean().item()


def calculate_multi_metrics(conf_mat):
    y_true = []
    y_pred = []
    for true_cls in range(4):
        for pred_cls in range(4):
            count = int(conf_mat[true_cls, pred_cls])
            y_true.extend([true_cls] * count)
            y_pred.extend([pred_cls] * count)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    total_correct = conf_mat.trace()
    total_samples = conf_mat.sum()
    accuracy = total_correct / (total_samples + 1e-8)
    macro_f1 = (f1[0] + f1[1] + f1[2] + f1[3]) / 4
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1
    }


def train_enhanced_model(model, train_loader, val_loader, optimizer, scheduler,
                         criterion, max_epochs, patience, snr):

    best_val_loss = float('inf')
    best_val_metrics = {
        'acc': 0.0, 'f1': 0.0, 'snr': 0.0, 'cc': 0.0, 'mse': float('inf')
    }
    early_stop_counter = 0
    best_model_path = f"dreamer_dualmamba_snr_{snr}_feedback.pth"

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        total_train_denoise = 0
        total_train_classify = 0
        train_correct = 0
        train_total = 0
        num_batches = 0

        for batch_data in train_loader:
            clean, noisy, labels = batch_data
            noisy, clean, labels = noisy.to(device), clean.to(device), labels.to(device)

            denoised, logits, early_logits, denoised_early_logits, _ = model(noisy, return_features=True)

            model.update_class_stats(labels)
            criterion.update_stats(labels)

            loss, loss_details = criterion(denoised, clean, logits, early_logits, denoised_early_logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN detected, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss_details['total'].item()
            total_train_denoise += loss_details['denoise'].item()
            total_train_classify += loss_details['classify'].item()
            num_batches += 1

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        if num_batches > 0:
            scheduler.step()
            avg_train_loss = total_train_loss / num_batches
            avg_train_denoise = total_train_denoise / num_batches
            avg_train_classify = total_train_classify / num_batches
            train_acc = train_correct / train_total if train_total > 0 else 0
        else:
            avg_train_loss = 0
            avg_train_denoise = 0
            avg_train_classify = 0
            train_acc = 0

        # 验证
        model.eval()
        total_val_loss = 0
        total_val_denoise = 0
        total_val_classify = 0
        total_val_snr = 0
        total_val_cc = 0
        val_correct = 0
        val_total = 0
        num_val_batches = 0
        conf_mat = np.zeros((4, 4), dtype=int)

        with torch.no_grad():
            for batch_data in val_loader:
                clean, noisy, labels = batch_data
                noisy, clean, labels = noisy.to(device), clean.to(device), labels.to(device)

                denoised, logits, early_logits, denoised_early_logits, _ = model(noisy, return_features=True)
                loss, loss_details = criterion(denoised, clean, logits, early_logits, denoised_early_logits, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_val_loss += loss_details['total'].item()
                total_val_denoise += loss_details['denoise'].item()
                total_val_classify += loss_details['classify'].item()
                total_val_snr += calculate_snr(clean, denoised)
                total_val_cc += calculate_cc(clean, denoised)
                num_val_batches += 1

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    conf_mat[true, pred] += 1

        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_denoise = total_val_denoise / num_val_batches
            avg_val_classify = total_val_classify / num_val_batches
            avg_val_snr = total_val_snr / num_val_batches
            avg_val_cc = total_val_cc / num_val_batches
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_metrics = calculate_multi_metrics(conf_mat)
            val_f1 = val_metrics['macro_f1']

            plateau_scheduler.step(avg_val_loss)
        else:
            avg_val_loss = float('inf')
            avg_val_denoise = 0
            avg_val_classify = 0
            avg_val_snr = 0
            avg_val_cc = 0
            val_acc = 0
            val_f1 = 0

        print(f"\nEpoch {epoch + 1}/{max_epochs} | SNR={snr}dB ")
        print(
            f"Train - Loss: {avg_train_loss:.4f} | Denoise: {avg_train_denoise:.4f} | Classify: {avg_train_classify:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f} | Denoise: {avg_val_denoise:.4f} | Classify: {avg_val_classify:.4f}")
        print(f"         SNR: {avg_val_snr:.2f}dB | CC: {avg_val_cc:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        current_score = 0.6 * val_acc + 0.4 * val_f1
        best_score = 0.6 * best_val_metrics['acc'] + 0.4 * best_val_metrics['f1']

        if current_score > best_score:
            best_val_metrics = {
                'acc': val_acc, 'f1': val_f1, 'snr': avg_val_snr,
                'cc': avg_val_cc, 'mse': avg_val_denoise
            }
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
            print(f"  Best model saved (Score: {current_score:.4f})")
        else:
            early_stop_counter += 1
            print(f"  Early stop: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    return best_val_metrics
