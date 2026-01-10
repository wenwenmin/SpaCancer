import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay


class ContrastiveLearningModule(nn.Module):
    """
    Contrastive Learning Module: Computes InfoNCE contrastive loss for a batch of
    image and gene features. This module aligns original features with multi-scale
    aggregated features.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)

    def forward(self, feat_i, feat_j):
        """
        Computes symmetric InfoNCE loss: (L_i2j + L_j2i) / 2

        Args:
            feat_i: Feature tensor i [Batch_size, Hidden_dim]
            feat_j: Feature tensor j [Batch_size, Hidden_dim]

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = feat_i.size(0)
        device = feat_i.device

        # Normalize feature vectors to the unit sphere
        feat_i = F.normalize(feat_i, p=2, dim=1)
        feat_j = F.normalize(feat_j, p=2, dim=1)

        # Compute similarity matrix [Batch_size, Batch_size]
        # Diagonal elements represent positive pairs
        sim_matrix = torch.matmul(feat_i, feat_j.T) / self.temperature

        # Labels: Diagonal elements are positives, others are negatives
        labels = torch.arange(batch_size, device=device)

        # Symmetric cross-entropy loss
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_i + loss_j) / 2

        return loss


class SpatialFeatureCrossAttention(nn.Module):
    """
    Spatial-Feature Cross-Attention Fusion Module.
    Fuses original features with spatially aggregated features using an attention mechanism.
    """

    def __init__(self, orig_dim, agg_dim, hidden_dim=512, num_heads=8):
        super().__init__()

        # Feature projection layers
        self.orig_proj = nn.Linear(orig_dim, hidden_dim)
        self.agg_proj = nn.Linear(agg_dim, hidden_dim)

        # Cross-attention module
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )
        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, orig_feat, agg_feat):
        # Feature projection
        orig_proj = self.orig_proj(orig_feat)
        agg_proj = self.agg_proj(agg_feat)

        # Cross-attention: Original features query aggregated features
        orig_attended, _ = self.cross_attn(
            orig_proj.unsqueeze(1),  # [B, 1, D]
            agg_proj.unsqueeze(1),  # [B, 1, D]
            agg_proj.unsqueeze(1)  # [B, 1, D]
        )
        orig_attended = orig_attended.squeeze(1)

        # Gated fusion
        combined = torch.cat([orig_proj, orig_attended], dim=1)
        gate_value = self.gate(combined)
        fused_feat = gate_value * orig_proj + (1 - gate_value) * orig_attended

        return self.output_proj(fused_feat)


class MultiScaleTransformerFusion(nn.Module):
    """
    Multi-Scale Transformer Fusion Module.
    Integrates features from different spatial scales using a Transformer encoder
    and a learnable gating mechanism.
    """

    def __init__(self, num_scales, hidden_dim):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim

        # Intra-scale feature refinement
        self.scale_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_scales)
        ])

        # Cross-scale relationship modeling
        self.cross_scale_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=2
        )

        # Global context (CLS) token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Gated fusion network
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * (num_scales + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=1)
        )

    def forward(self, scale_features):
        """
        Args:
            scale_features: List of tensors [feat_scale1, feat_scale2, ...],
                            each with shape [B, hidden_dim].
        Returns:
            fused_feat: [B, hidden_dim]
        """
        # 1. Refine features for each scale
        refined = []
        for i, feat in enumerate(scale_features):
            if feat.size(-1) != self.hidden_dim:
                feat = feat[:, :self.hidden_dim]
            refined.append(self.scale_refiners[i](feat))

        # 2. Stack scales [B, num_scales, D]
        scale_tensor = torch.stack(refined, dim=1)

        # 3. Add CLS token
        batch_size = scale_tensor.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        transformer_input = torch.cat([cls_tokens, scale_tensor], dim=1)  # [B, num_scales+1, D]

        # 4. Cross-scale attention
        attn_output = self.cross_scale_attn(transformer_input)

        # 5. Separate CLS token and scale features
        cls_output = attn_output[:, 0]  # [B, D]
        scale_output = attn_output[:, 1:]  # [B, num_scales, D]

        # 6. Prepare gate input (Flatten features)
        gate_input = torch.cat([
            cls_output,
            scale_output.flatten(start_dim=1)
        ], dim=1)  # [B, (num_scales + 1) * D]

        # 7. Compute gate weights [B, num_scales]
        gate_weights = self.fusion_gate(gate_input)

        # 8. Weighted fusion
        weighted = scale_output * gate_weights.unsqueeze(-1)
        fused_feat = torch.sum(weighted, dim=1)  # [B, D]

        # 9. Combine with CLS token context
        return fused_feat + cls_output


class BidirectionalCrossModalFusion(nn.Module):
    """
    Bidirectional Cross-Modal Fusion Module.
    Uses bidirectional attention to fuse image and gene expression features.
    """

    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Modality projection
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gene_proj = nn.Linear(hidden_dim, hidden_dim)

        # Bidirectional Cross-Attention
        self.img_to_gene_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gene_to_img_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feature Refinement
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, img_feat, gene_feat):
        """
        Args:
            img_feat: [B, H]
            gene_feat: [B, H]
        Returns:
            Fused feature [B, H]
        """
        # 1. Project modalities
        img_proj = self.img_proj(img_feat).unsqueeze(1)  # [B, 1, H]
        gene_proj = self.gene_proj(gene_feat).unsqueeze(1)  # [B, 1, H]

        # 2. Bidirectional Attention
        # Image attends to Gene
        img_attended, _ = self.img_to_gene_attn(
            query=img_proj, key=gene_proj, value=gene_proj
        )

        # Gene attends to Image
        gene_attended, _ = self.gene_to_img_attn(
            query=gene_proj, key=img_proj, value=img_proj
        )

        img_attended = img_attended.squeeze(1)  # [B, H]
        gene_attended = gene_attended.squeeze(1)  # [B, H]

        # 3. Concatenate and Refine
        fused = torch.cat([img_attended, gene_attended], dim=-1)  # [B, 2H]
        return self.refiner(fused)


class RobustGatedClassifier(nn.Module):
    """
    Robust Gated Classifier: Uses residual gated blocks for feature classification.
    """

    def __init__(self, input_dim, base_dim=1024, num_blocks=4, dropout=0.4, return_binary=False):
        super().__init__()
        self.return_binary = return_binary

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Gated Residual Blocks
        self.res_blocks = nn.ModuleList()
        self.gates = nn.ModuleList()

        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(base_dim, base_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(base_dim * 2, base_dim),
                nn.LayerNorm(base_dim),
                nn.GELU()
            )
            self.res_blocks.append(block)

            gate = nn.Sequential(
                nn.Linear(base_dim, 1),
                nn.Sigmoid()
            )
            self.gates.append(gate)

        # Feature Compression
        self.compress = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.LayerNorm(base_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(base_dim // 2, base_dim // 4),
            nn.GELU(),
            nn.Linear(base_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, threshold=0.2):
        # Initial projection
        x = self.input_proj(x)

        # Gated residual processing
        for block, gate in zip(self.res_blocks, self.gates):
            identity = x
            residual = block(x)
            g = gate(x)
            # Gated residual connection
            x = identity * g + residual * (1 - g)

        # Compression and Classification
        x = self.compress(x)
        prob = self.classifier(x)

        if self.return_binary:
            return (prob >= threshold).float()
        else:
            return prob


class SpatialCancerClassifier(nn.Module):
    """
    Advanced Spatial-Aware Cancer Classifier with Contrastive Learning.
    Integrates multi-scale spatial features and multi-modal data.
    """

    def __init__(self, img_dim, gene_dim, scales=[0], hidden_dim=512, directions=8,
                 cl_temperature=0.07, cl_weight=0.1,
                 focal_alpha=0.25, focal_gamma=2.0,
                 dice_epsilon=1e-5, dice_weight=1.0):
        super().__init__()
        self.scales = scales
        self.cl_weight = cl_weight * 1 / len(scales)

        # Loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_epsilon = dice_epsilon
        self.dice_weight = dice_weight

        # 1. Original Feature Encoders
        self.orig_img_encoder = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )
        self.orig_gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )

        # 2. Multi-Scale Fusion Modules
        self.img_fusion_modules = nn.ModuleList()
        self.gene_fusion_modules = nn.ModuleList()
        img_agg_dim = img_dim + directions
        gene_agg_dim = gene_dim + directions

        for _ in scales:
            self.img_fusion_modules.append(
                SpatialFeatureCrossAttention(hidden_dim, img_agg_dim, hidden_dim)
            )
            self.gene_fusion_modules.append(
                SpatialFeatureCrossAttention(hidden_dim, gene_agg_dim, hidden_dim)
            )

        # 3. Multi-Scale Transformer Fusion
        self.img_scale_fusion = MultiScaleTransformerFusion(len(scales) + 1, hidden_dim)
        self.gene_scale_fusion = MultiScaleTransformerFusion(len(scales) + 1, hidden_dim)

        # 4. Cross-Modal Fusion
        self.cross_modal_fusion = BidirectionalCrossModalFusion(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=0.4
        )

        # 5. Prediction Head
        self.predictor = RobustGatedClassifier(
            input_dim=hidden_dim,
            base_dim=512,
            num_blocks=4,
            dropout=0.6
        )

        # 6. Contrastive Learning Modules
        # Separate CL module for each scale including original
        self.cl_modules = nn.ModuleList([
            ContrastiveLearningModule(temperature=cl_temperature)
            for _ in range(len(scales) + 1)
        ])

    def forward(self, inputs, labels=None):
        """
        Forward pass.

        Args:
            inputs: Dictionary containing input tensors.
            labels: Ground truth labels (optional).

        Returns:
            output: Prediction probabilities.
            total_loss: (Optional) If labels are provided.
            classification_loss: (Optional) BCE loss.
            total_cl_loss: (Optional) Contrastive loss.
        """
        orig_img = inputs['orig_img']
        orig_gene = inputs['orig_gene']

        # 1. Encode original features
        img_feat = self.orig_img_encoder(orig_img)  # [B, H]
        gene_feat = self.orig_gene_encoder(orig_gene)  # [B, H]

        # 2. Contrastive loss for original features
        cl_loss_orig = self.cl_modules[0](img_feat, gene_feat)
        total_cl_loss = cl_loss_orig

        # 3. Multi-scale aggregation and fusion
        img_scale_feats = [img_feat]
        gene_scale_feats = [gene_feat]

        for i, scale in enumerate(self.scales):
            agg_img = inputs[f'agg_img_{scale}']
            agg_gene = inputs[f'agg_gene_{scale}']

            # Fuse current scale aggregated features
            fused_img_scale = self.img_fusion_modules[i](img_feat, agg_img)
            fused_gene_scale = self.gene_fusion_modules[i](gene_feat, agg_gene)

            img_scale_feats.append(fused_img_scale)
            gene_scale_feats.append(fused_gene_scale)

            # Contrastive loss for current scale
            cl_loss_scale = self.cl_modules[i + 1](fused_img_scale, fused_gene_scale)
            total_cl_loss = total_cl_loss + cl_loss_scale

        # 4. Multi-Scale Transformer Fusion
        img_fused = self.img_scale_fusion(img_scale_feats)  # [B, H]
        gene_fused = self.gene_scale_fusion(gene_scale_feats)  # [B, H]

        # 5. Cross-Modal Fusion
        fused_all = self.cross_modal_fusion(img_fused, gene_fused)  # [B, H]

        # 6. Prediction
        output = self.predictor(fused_all)

        # 7. Loss Computation
        if labels is not None:
            # Classification Loss (BCE)
            if labels.dim() > 2:
                labels = labels.squeeze(-1)

            classification_loss = F.binary_cross_entropy(output, labels.float())
            total_loss = classification_loss + self.cl_weight * total_cl_loss
            return output, total_loss, classification_loss, total_cl_loss
        else:
            return output, total_cl_loss



