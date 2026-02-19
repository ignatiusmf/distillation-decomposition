"""
Distillation methods for knowledge transfer from teacher to student networks.

Each distillation method inherits from DistillationMethod base class and implements:
- extra_loss(): computes the distillation loss given teacher/student outputs
- Any required modules (e.g., connector layers for factor transfer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Optional


class DistillationMethod(ABC):
    """Base class for all distillation methods."""
    
    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for distillation loss. Final loss = (1-alpha)*CE + alpha*distill_loss
        """
        self.alpha = alpha
    
    @abstractmethod
    def extra_loss(
        self, 
        student_outputs: List[torch.Tensor], 
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distillation loss.
        
        Args:
            student_outputs: List of [layer1, layer2, layer3, logits] from student
            teacher_outputs: List of [layer1, layer2, layer3, logits] from teacher
            targets: Ground truth labels
            
        Returns:
            Distillation loss tensor
        """
        pass
    
    def get_trainable_modules(self) -> List[nn.Module]:
        """Return any extra trainable modules (e.g., connectors) for the optimizer."""
        return []
    
    def to(self, device: str) -> 'DistillationMethod':
        """Move any internal modules to device."""
        return self


class NoDistillation(DistillationMethod):
    """Dummy distillation method for pure training without a teacher."""
    
    def __init__(self):
        super().__init__(alpha=0.0)
    
    def extra_loss(
        self, 
        student_outputs: List[torch.Tensor], 
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(0.0)


class LogitDistillation(DistillationMethod):
    """
    Logit-based knowledge distillation (Hinton et al., 2015).
    
    Softens the teacher's output probabilities with temperature and trains
    the student to match these soft targets via KL divergence.
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        """
        Args:
            alpha: Weight for distillation loss
            temperature: Softmax temperature for softening distributions
        """
        super().__init__(alpha)
        self.temperature = temperature
    
    def extra_loss(
        self, 
        student_outputs: List[torch.Tensor], 
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        student_logits = student_outputs[-1]  # Last element is logits
        teacher_logits = teacher_outputs[-1]
        
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss (scaled by T^2 as per Hinton et al.)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return kl_loss


class FactorTransfer(DistillationMethod):
    """
    Factor Transfer (Kim et al., 2018).
    
    Transfers knowledge through paraphraser-translator architecture.
    The paraphraser extracts factors from teacher features, and the 
    translator learns to match these factors from student features.
    """
    
    def __init__(
        self, 
        student_channels: List[int],
        teacher_channels: List[int],
        alpha: float = 0.5,
        factor_dim: int = 64,
        layers_to_use: List[int] = [0, 1, 2]  # Which intermediate layers to use
    ):
        """
        Args:
            student_channels: Channel dimensions for each student layer [16, 32, 64]
            teacher_channels: Channel dimensions for each teacher layer [16, 32, 64]
            alpha: Weight for distillation loss
            factor_dim: Dimension of the factor space
            layers_to_use: Which layers (0=layer1, 1=layer2, 2=layer3) to transfer
        """
        super().__init__(alpha)
        self.factor_dim = factor_dim
        self.layers_to_use = layers_to_use
        
        # Create paraphrasers (teacher -> factors) and translators (student -> factors)
        self.paraphrasers = nn.ModuleList()
        self.translators = nn.ModuleList()
        
        for i in layers_to_use:
            # Paraphraser: teacher features -> factors
            self.paraphrasers.append(
                nn.Sequential(
                    nn.Conv2d(teacher_channels[i], factor_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(factor_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(factor_dim, factor_dim, kernel_size=1, bias=False),
                )
            )
            
            # Translator: student features -> factors
            self.translators.append(
                nn.Sequential(
                    nn.Conv2d(student_channels[i], factor_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(factor_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(factor_dim, factor_dim, kernel_size=1, bias=False),
                )
            )
    
    def extra_loss(
        self, 
        student_outputs: List[torch.Tensor], 
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0.0
        
        for idx, layer_idx in enumerate(self.layers_to_use):
            student_feat = student_outputs[layer_idx]
            teacher_feat = teacher_outputs[layer_idx]
            
            # Handle spatial dimension mismatch via adaptive pooling
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                target_size = (teacher_feat.shape[2], teacher_feat.shape[3])
                student_feat = F.adaptive_avg_pool2d(student_feat, target_size)
            
            # Extract factors
            teacher_factor = self.paraphrasers[idx](teacher_feat)
            student_factor = self.translators[idx](student_feat)
            
            # Normalize factors (L2 norm along channel dimension)
            teacher_factor = F.normalize(teacher_factor, p=2, dim=1)
            student_factor = F.normalize(student_factor, p=2, dim=1)
            
            # L2 loss between factors
            loss = F.mse_loss(student_factor, teacher_factor.detach())
            total_loss = total_loss + loss
        
        return total_loss / len(self.layers_to_use)  # type: ignore
    
    def get_trainable_modules(self) -> List[nn.Module]:
        """Return paraphrasers and translators for optimizer."""
        return [self.paraphrasers, self.translators]
    
    def to(self, device: str) -> 'FactorTransfer':
        """Move modules to device."""
        self.paraphrasers = self.paraphrasers.to(device)
        self.translators = self.translators.to(device)
        return self


class AttentionTransfer(DistillationMethod):
    """
    Attention Transfer (Zagoruyko & Komodakis, 2017).

    Transfers spatial attention maps derived from intermediate feature maps.
    Attention map = channel-wise sum of squared activations, L2-normalized.
    """

    def __init__(self, alpha: float = 0.5, layers_to_use: List[int] = [0, 1, 2]):
        super().__init__(alpha)
        self.layers_to_use = layers_to_use

    def _attention_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention: sum of squares across channels, then L2-normalize."""
        # feature_map: (B, C, H, W) -> (B, H*W)
        am = (feature_map ** 2).sum(dim=1)  # (B, H, W)
        am = am.view(am.size(0), -1)  # (B, H*W)
        am = F.normalize(am, p=2, dim=1)
        return am

    def extra_loss(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0.0
        for layer_idx in self.layers_to_use:
            s_feat = student_outputs[layer_idx]
            t_feat = teacher_outputs[layer_idx]

            s_am = self._attention_map(s_feat)
            t_am = self._attention_map(t_feat)

            # Handle spatial dimension mismatch
            if s_am.shape != t_am.shape:
                s_side = int(s_am.shape[1] ** 0.5)
                t_side = int(t_am.shape[1] ** 0.5)
                s_am_2d = s_am.view(s_am.size(0), 1, s_side, s_side)
                s_am_2d = F.adaptive_avg_pool2d(s_am_2d, (t_side, t_side))
                s_am = s_am_2d.view(s_am.size(0), -1)
                s_am = F.normalize(s_am, p=2, dim=1)

            total_loss = total_loss + (s_am - t_am.detach()).pow(2).sum(dim=1).mean()

        return total_loss / len(self.layers_to_use)


class FitNets(DistillationMethod):
    """
    FitNets (Romero et al., 2015).

    Matches intermediate representations via a learned regressor (connector)
    that projects the student's feature map to the teacher's dimensionality.
    """

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        alpha: float = 0.5,
        hint_layer: int = 1,
    ):
        super().__init__(alpha)
        self.hint_layer = hint_layer

        # 1x1 conv to match channel dimensions
        self.connector = nn.Conv2d(
            student_channels[hint_layer],
            teacher_channels[hint_layer],
            kernel_size=1, bias=False
        )

    def extra_loss(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        s_feat = student_outputs[self.hint_layer]
        t_feat = teacher_outputs[self.hint_layer]

        s_proj = self.connector(s_feat)

        # Handle spatial dimension mismatch
        if s_proj.shape[2:] != t_feat.shape[2:]:
            s_proj = F.adaptive_avg_pool2d(s_proj, t_feat.shape[2:])

        return F.mse_loss(s_proj, t_feat.detach())

    def get_trainable_modules(self) -> List[nn.Module]:
        return [self.connector]

    def to(self, device: str) -> 'FitNets':
        self.connector = self.connector.to(device)
        return self


class RKD(DistillationMethod):
    """
    Relational Knowledge Distillation (Park et al., 2019).

    Transfers pairwise distance relations and angle relations between
    sample embeddings rather than individual outputs.
    Uses the GAP'd final feature map (layer3) as the embedding.
    """

    def __init__(self, alpha: float = 0.5, distance_weight: float = 25.0, angle_weight: float = 50.0):
        super().__init__(alpha)
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    def _pdist(self, e: torch.Tensor) -> torch.Tensor:
        """Pairwise Euclidean distance, normalized by mean."""
        dot = e @ e.t()
        sq_norms = dot.diagonal()
        dist = (sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2 * dot).clamp(min=1e-12).sqrt()
        mean_dist = dist[dist > 0].mean().detach()
        return dist / mean_dist

    def _angle(self, e: torch.Tensor) -> torch.Tensor:
        """Pairwise angle relations (cosine of angle in triplets)."""
        # e: (B, D)
        e_norm = F.normalize(e, p=2, dim=1)
        return e_norm @ e_norm.t()

    def extra_loss(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Use final feature map (layer3), GAP to get embeddings
        s_feat = student_outputs[2]  # (B, C, H, W)
        t_feat = teacher_outputs[2]

        s_emb = F.adaptive_avg_pool2d(s_feat, 1).view(s_feat.size(0), -1)
        t_emb = F.adaptive_avg_pool2d(t_feat, 1).view(t_feat.size(0), -1)

        # Distance-wise loss
        loss_d = F.smooth_l1_loss(self._pdist(s_emb), self._pdist(t_emb).detach())

        # Angle-wise loss
        loss_a = F.smooth_l1_loss(self._angle(s_emb), self._angle(t_emb).detach())

        return self.distance_weight * loss_d + self.angle_weight * loss_a


class NST(DistillationMethod):
    """
    Neuron Selectivity Transfer (Huang & Wang, 2017).

    Matches the distribution of neuron selectivity patterns between
    teacher and student via MMD (Maximum Mean Discrepancy) on
    activation distributions.
    """

    def __init__(self, alpha: float = 0.5, layers_to_use: List[int] = [0, 1, 2]):
        super().__init__(alpha)
        self.layers_to_use = layers_to_use

    def _nst_loss(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Compute NST loss: MMD between channel-normalized activation distributions."""
        # Reshape to (B, C, H*W) and L2-normalize across spatial dims
        s = s_feat.view(s_feat.size(0), s_feat.size(1), -1)  # (B, C, N)
        t = t_feat.view(t_feat.size(0), t_feat.size(1), -1)

        s = F.normalize(s, p=2, dim=2)
        t = F.normalize(t, p=2, dim=2)

        # Gram matrices as kernel: (B, C, C)
        s_gram = torch.bmm(s, s.transpose(1, 2))
        t_gram = torch.bmm(t, t.transpose(1, 2))

        return F.mse_loss(s_gram, t_gram.detach())

    def extra_loss(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0.0
        for layer_idx in self.layers_to_use:
            s_feat = student_outputs[layer_idx]
            t_feat = teacher_outputs[layer_idx]

            # Handle spatial dimension mismatch
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])

            total_loss = total_loss + self._nst_loss(s_feat, t_feat)

        return total_loss / len(self.layers_to_use)


# Registry of available distillation methods
DISTILLATION_METHODS = {
    'none': NoDistillation,
    'logit': LogitDistillation,
    'factor_transfer': FactorTransfer,
    'attention_transfer': AttentionTransfer,
    'fitnets': FitNets,
    'rkd': RKD,
    'nst': NST,
}


def get_distillation_method(
    method_name: str,
    student_model: nn.Module,
    teacher_model: Optional[nn.Module] = None,
    **kwargs
) -> DistillationMethod:
    """
    Factory function to create distillation method instances.
    
    Args:
        method_name: Name of the distillation method ('none', 'logit', 'factor_transfer')
        student_model: The student model
        teacher_model: The teacher model (None for pure training)
        **kwargs: Additional arguments for the specific method
        
    Returns:
        DistillationMethod instance
    """
    if method_name not in DISTILLATION_METHODS:
        raise ValueError(f"Unknown distillation method: {method_name}. "
                        f"Available: {list(DISTILLATION_METHODS.keys())}")
    
    if method_name == 'none':
        return NoDistillation()
    
    if method_name == 'logit':
        return LogitDistillation(
            alpha=kwargs.get('alpha', 0.5),
            temperature=kwargs.get('temperature', 4.0)
        )
    
    if method_name == 'factor_transfer':
        # For ResNet_simple models: channels are [16, 32, 64] for all variants
        student_channels = [16, 32, 64]
        teacher_channels = [16, 32, 64]

        return FactorTransfer(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            alpha=kwargs.get('alpha', 0.5),
            factor_dim=kwargs.get('factor_dim', 64),
            layers_to_use=kwargs.get('layers_to_use', [0, 1, 2])
        )

    if method_name == 'attention_transfer':
        return AttentionTransfer(
            alpha=kwargs.get('alpha', 0.5),
            layers_to_use=kwargs.get('layers_to_use', [0, 1, 2])
        )

    if method_name == 'fitnets':
        student_channels = [16, 32, 64]
        teacher_channels = [16, 32, 64]

        return FitNets(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            alpha=kwargs.get('alpha', 0.5),
            hint_layer=kwargs.get('hint_layer', 1),
        )

    if method_name == 'rkd':
        return RKD(
            alpha=kwargs.get('alpha', 0.5),
            distance_weight=kwargs.get('distance_weight', 25.0),
            angle_weight=kwargs.get('angle_weight', 50.0),
        )

    if method_name == 'nst':
        return NST(
            alpha=kwargs.get('alpha', 0.5),
            layers_to_use=kwargs.get('layers_to_use', [0, 1, 2])
        )

    raise ValueError(f"Method {method_name} not implemented in factory")
