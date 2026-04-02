import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba


class StdPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1)
        std = torch.sqrt(var + 1e-8)
        return std.unsqueeze(-1)


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.enhance = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels // 4),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, L = x.shape
        avg_feat = self.avg_pool(x).view(B, C)
        avg_out = self.mlp(avg_feat)
        max_feat = self.max_pool(x).view(B, C)
        max_out = self.mlp(max_feat)
        attn_weights = self.sigmoid(avg_out + max_out).view(B, C, 1)
        x_attended = x * attn_weights
        x_enhanced = self.enhance(x_attended)
        x_out = x_attended + 0.1 * x_enhanced
        return x_out


class TimeOnlyMambaBlock(nn.Module):

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.time_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.temp_enhance = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.time_mamba(x)
        x = self.norm1(x)
        x = self.norm2(residual + self.dropout(x))
        x_conv = x.transpose(1, 2)
        x_conv = self.temp_enhance(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + self.dropout(x_conv)
        return x


class EnhancedFreqFeature(nn.Module):

    def __init__(self, in_channels=32, d_model=128):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        self.n_fft_scales = [32, 64, 128]
        scale_dims = [43, 43, 42]

        self.freq_projs = nn.ModuleDict()
        for idx, n_fft in enumerate(self.n_fft_scales):
            self.freq_projs[str(n_fft)] = nn.Sequential(
                nn.Conv1d(min(in_channels, 16) * 2, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, scale_dims[idx])
            )

        self.band_energy = nn.Sequential(
            nn.Linear(in_channels * 5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, L = x.shape
        freq_feats = []

        for idx, n_fft in enumerate(self.n_fft_scales):
            n_ch = min(C, 16)
            x_sel = x[:, :n_ch, :]
            channel_feats = []
            for c in range(n_ch):
                fft = torch.fft.rfft(x_sel[:, c, :], n=n_fft, dim=-1)
                mag = torch.abs(fft)
                phase = torch.angle(fft)
                channel_feats.append(torch.stack([mag, phase], dim=1))
            if channel_feats:
                feat = torch.stack(channel_feats, dim=2)
                B, _, n_ch, n_freq = feat.shape
                feat = feat.view(B, 2 * n_ch, n_freq)
                proj_feat = self.freq_projs[str(n_fft)](feat)
                freq_feats.append(proj_feat)

        if freq_feats:
            combined_freq = torch.cat(freq_feats, dim=-1)
        else:
            combined_freq = torch.zeros(B, self.d_model).to(x.device)

        fft_all = torch.fft.rfft(x, n=128, dim=-1)
        mag_all = torch.abs(fft_all)
        freqs = torch.linspace(0, 64, mag_all.size(2)).to(x.device)

        bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 45)
        }

        band_energies = []
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_energy = mag_all[:, :, mask].pow(2).mean(dim=-1)
            band_energies.append(band_energy)

        band_feat = torch.cat(band_energies, dim=-1)
        band_feat = self.band_energy(band_feat)
        freq_feat = combined_freq + band_feat
        freq_feat = self.final_norm(freq_feat)

        return freq_feat


class EnhancedDualDomainMambaBlock(nn.Module):

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.time_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.freq_gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.temp_enhance = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm1d(d_model), nn.GELU()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freq_feat=None):
        residual = x
        x = self.time_mamba(x)
        x = self.norm1(x)
        if freq_feat is not None:
            gate = self.freq_gate(freq_feat).unsqueeze(1)
            x = x * (1 + gate)
        x = self.norm2(residual + self.dropout(x))

        if freq_feat is not None:
            freq_seq = freq_feat.unsqueeze(1).expand(-1, x.size(1), -1)
            attn_out, _ = self.cross_attn(freq_seq, x, x)
            x = x + self.dropout(attn_out)
            x = self.norm3(x)

        x_conv = x.transpose(1, 2)
        x_conv = self.temp_enhance(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + self.dropout(x_conv)
        return x


class DenoisedSignalFeatureExtractor(nn.Module):

    def __init__(self, in_channels=32, d_model=128, num_mamba_layers=2):
        super().__init__()

        self.single_scale_proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)

        self.mamba_blocks = nn.ModuleList([
            TimeOnlyMambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2, dropout=0.1)
            for _ in range(num_mamba_layers)
        ])

        self.channel_attention = ChannelAttention(d_model, reduction=8)
        self.att_enhance = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU()
        )
        self.temporal_pooling = nn.ModuleDict({
            'mean': nn.AdaptiveAvgPool1d(1),
            'max': nn.AdaptiveMaxPool1d(1),
            'std': StdPool()
        })
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, L = x.shape

        time_feat = self.single_scale_proj(x)
        time_feat = time_feat.transpose(1, 2)

        if L == 128:
            time_feat = time_feat + self.pos_encoding[:, :L, :]
        else:
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), size=L, mode='linear', align_corners=False
            ).transpose(1, 2)
            time_feat = time_feat + pos_enc

        for block in self.mamba_blocks:
            time_feat = block(time_feat)

        time_feat_t = time_feat.transpose(1, 2)
        time_feat_attended = self.channel_attention(time_feat_t)
        time_feat_enhanced = self.att_enhance(time_feat_attended) + time_feat_attended

        global_feat = self.temporal_pooling['mean'](time_feat_enhanced).squeeze(-1)

        pooled_features = []
        for name, pool in self.temporal_pooling.items():
            pooled = pool(time_feat_enhanced)
            pooled_features.append(pooled.squeeze(-1))

        time_pooled = torch.cat(pooled_features, dim=1)
        return time_pooled, global_feat, time_feat_enhanced


class NoMultiScaleDualDomainEncoder(nn.Module):

    def __init__(self, in_channels=32, length=128, d_model=128, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.length = length

        self.time_proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        self.freq_extractor = EnhancedFreqFeature(in_channels, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, length, d_model) * 0.02)

        self.mamba_blocks = nn.ModuleList([
            EnhancedDualDomainMambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, L = x.shape
        time_feat = self.time_proj(x)
        time_feat = time_feat.transpose(1, 2)

        if L == self.length:
            time_feat = time_feat + self.pos_encoding
        else:
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), size=L, mode='linear', align_corners=False
            ).transpose(1, 2)
            time_feat = time_feat + pos_enc

        freq_feat = self.freq_extractor(x)

        for block in self.mamba_blocks:
            time_feat = block(time_feat, freq_feat)

        time_feat_t = time_feat.transpose(1, 2)
        global_feat = self.global_pool(time_feat_t).squeeze(-1)
        local_feat = self.local_conv(time_feat_t)
        time_feat = (time_feat + local_feat.transpose(1, 2)) / 2
        time_feat = self.norm(time_feat)

        return time_feat, freq_feat, global_feat


class EnhancedDualDomainJointModel(nn.Module):

    def __init__(self, in_channels=32, length=128, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.length = length
        self.d_model = 128

        self.input_norm = nn.BatchNorm1d(in_channels)

        self.encoder_noisy = NoMultiScaleDualDomainEncoder(
            in_channels=in_channels, length=length, d_model=self.d_model, num_layers=3
        )

        self.denoise_decoder = nn.Sequential(
            nn.ConvTranspose1d(self.d_model, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=3, padding=1)
        )

        self.denoised_feature_extractor = DenoisedSignalFeatureExtractor(
            in_channels=in_channels, d_model=self.d_model, num_mamba_layers=2
        )

        self.channel_attention_noisy = ChannelAttention(self.d_model, reduction=8)
        self.att_enhance_noisy = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model), nn.GELU()
        )

        self.temporal_pooling_noisy = nn.ModuleDict({
            'mean': nn.AdaptiveAvgPool1d(1),
            'max': nn.AdaptiveMaxPool1d(1),
            'std': StdPool()
        })

        self.feature_fusion = nn.Sequential(
            nn.Linear(5 * self.d_model + 4 * self.d_model, 1024),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 5 * self.d_model)
        )

        self.classifier = nn.Sequential(
            nn.Linear(5 * self.d_model, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.class_attention = nn.Parameter(torch.ones(num_classes))
        self.register_buffer('class_stats', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0))

        self.early_classifier = nn.Sequential(
            nn.Linear(self.d_model, 64), nn.GELU(), nn.Linear(64, num_classes)
        )

        self.denoised_early_classifier = nn.Sequential(
            nn.Linear(self.d_model, 64), nn.GELU(), nn.Linear(64, num_classes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1 and not isinstance(p, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.xavier_uniform_(p)

    def update_class_stats(self, labels):
        with torch.no_grad():
            unique_classes, counts = torch.unique(labels, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                self.class_stats[cls] += count
            self.total_samples += labels.size(0)

    def get_class_weights(self):
        if self.total_samples == 0:
            return torch.ones(self.num_classes).to(device)
        class_freq = self.class_stats / (self.total_samples + 1e-8)
        inverse_weights = 1.0 / (class_freq + 1e-8)
        inverse_weights = inverse_weights / inverse_weights.sum()
        adaptive_weights = inverse_weights * F.softmax(self.class_attention, dim=0)
        return adaptive_weights

    def _extract_noisy_classification_features(self, encoder_output):
        time_feat, freq_feat, global_feat = encoder_output
        time_feat_t = time_feat.transpose(1, 2)
        time_feat_attended = self.channel_attention_noisy(time_feat_t)
        time_feat_enhanced = self.att_enhance_noisy(time_feat_attended) + time_feat_attended

        pooled_features = []
        for name, pool in self.temporal_pooling_noisy.items():
            pooled = pool(time_feat_enhanced)
            pooled_features.append(pooled.squeeze(-1))

        time_pooled = torch.cat(pooled_features, dim=1)
        return time_pooled, freq_feat, global_feat

    def forward(self, x, return_features=False):
        B, C, _, L = x.shape
        x_squeezed = x.squeeze(2)
        x_norm = self.input_norm(x_squeezed)

        noisy_time_feat, noisy_freq_feat, noisy_global_feat = self.encoder_noisy(x_norm)
        early_logits = self.early_classifier(noisy_global_feat)

        denoise_input = noisy_time_feat.transpose(1, 2)
        denoised = self.denoise_decoder(denoise_input)
        denoised = denoised + x_norm
        denoised_out = denoised.unsqueeze(2)

        denoised_pooled, denoised_global_feat, _ = self.denoised_feature_extractor(denoised)
        noisy_pooled, noisy_freq_feat, noisy_global_feat = self._extract_noisy_classification_features(
            (noisy_time_feat, noisy_freq_feat, noisy_global_feat)
        )

        denoised_early_logits = self.denoised_early_classifier(denoised_global_feat)

        noisy_combined = torch.cat([noisy_pooled, noisy_freq_feat, noisy_global_feat], dim=1)
        denoised_combined = torch.cat([denoised_pooled, denoised_global_feat], dim=1)
        fused_features = torch.cat([noisy_combined, denoised_combined], dim=1)
        fused_features = self.feature_fusion(fused_features)
        logits = self.classifier(fused_features)

        if return_features:
            return denoised_out, logits, early_logits, denoised_early_logits, (
                noisy_time_feat, noisy_freq_feat, noisy_global_feat, denoised_global_feat, denoised_global_feat)

        return denoised_out, logits


class EnhancedLoss(nn.Module):

    def __init__(self, denoise_weight=3.0, classify_weight=1.0, num_classes=4):
        super().__init__()
        self.denoise_weight = denoise_weight
        self.classify_weight = classify_weight
        self.num_classes = num_classes

        self.denoise_criterion = nn.MSELoss(reduction='none')
        self.classify_criterion = nn.CrossEntropyLoss(reduction='none')

        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0))

    def update_stats(self, targets):
        with torch.no_grad():
            for i in range(self.num_classes):
                self.class_counts[i] += (targets == i).sum().item()
            self.total_samples += targets.size(0)

    def get_class_weights(self, targets):
        if self.total_samples > 0:
            class_freq = self.class_counts / (self.total_samples + 1e-8)
            weights = 1.0 / (class_freq + 1e-8)
            weights = weights / weights.sum() * self.num_classes
            return weights[targets]
        return torch.ones_like(targets).float()

    def compute_confidence(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = math.log(self.num_classes)
        confidence = 1.0 - entropy / max_entropy
        return confidence

    def forward(self, denoised, clean, logits, early_logits, denoised_early_logits, labels):
        B = denoised.size(0)

        per_sample_denoise = self.denoise_criterion(denoised, clean)
        per_sample_denoise = per_sample_denoise.reshape(B, -1).mean(dim=1)

        class_weights = self.get_class_weights(labels).to(labels.device)
        per_sample_classify = self.classify_criterion(logits, labels) * class_weights
        per_sample_early = self.classify_criterion(early_logits, labels) * class_weights
        per_sample_denoised_early = self.classify_criterion(denoised_early_logits, labels) * class_weights

        confidence = self.compute_confidence(logits).detach()

        denoise_weights = 1.0 - 0.5 * confidence
        classify_weights = 1.0 + 0.5 * (1.0 - confidence)

        weighted_denoise = (per_sample_denoise * denoise_weights).mean()
        weighted_classify = (per_sample_classify * classify_weights).mean()
        weighted_early = (per_sample_early * classify_weights).mean()
        weighted_denoised_early = (per_sample_denoised_early * classify_weights).mean()

        total_loss = (self.denoise_weight * weighted_denoise +
                      self.classify_weight * weighted_classify )

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        loss_details = {
            'total': total_loss, 'denoise': weighted_denoise, 'classify': weighted_classify,
            'early_classify': weighted_early, 'denoised_early': weighted_denoised_early,
            'confidence': confidence.mean(), 'accuracy': accuracy
        }

        return total_loss, loss_details
