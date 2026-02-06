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


# Registry of available distillation methods
DISTILLATION_METHODS = {
    'none': NoDistillation,
    'logit': LogitDistillation,
    'factor_transfer': FactorTransfer,
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
    
    raise ValueError(f"Method {method_name} not implemented in factory")
