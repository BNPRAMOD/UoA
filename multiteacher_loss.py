
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def normalize_lambdas(lmb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if lmb.dim() == 1:
        return lmb / lmb.sum().clamp_min(eps)
    return lmb / lmb.sum(dim=-1, keepdim=True).clamp_min(eps)


def fuse_logits(
    teacher_logits: Dict[str, torch.Tensor],
    teacher_order: List[str],
    lambdas: torch.Tensor
) -> torch.Tensor:
    logits_list = [teacher_logits[k] for k in teacher_order]
    stacked = torch.stack(logits_list, dim=1)  # (B, T, C)

    lambdas = normalize_lambdas(lambdas).to(stacked.device)
    if lambdas.dim() == 1:
        lambdas = lambdas.unsqueeze(0).expand(stacked.size(0), -1)  # (B, T)

    return (stacked * lambdas.unsqueeze(-1)).sum(dim=1)  # (B, C)


def kd_soft(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def kd_hard(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student_logits, teacher_logits.argmax(dim=-1))


class FrozenTeacherEnsemble(nn.Module):
    def __init__(self, teacher_names: List[str], device: torch.device):
        super().__init__()
        self.models = nn.ModuleDict({
            name: timm.create_model(name, pretrained=True, num_classes=1000).eval().to(device)
            for name in teacher_names
        })
        for m in self.models.values():
            for p in m.parameters():
                p.requires_grad_(False)
        self.teacher_order = list(self.models.keys())

    @torch.no_grad()
    def forward(self, x):
        return {k: m(x) for k, m in self.models.items()}


class TeacherLogitAdapter(nn.Module):
    def __init__(self, teacher_keys: List[str], student_num_classes: int):
        super().__init__()
        self.adapters = nn.ModuleDict({
            k: nn.Linear(1000, student_num_classes, bias=False) for k in teacher_keys
        })

    def forward(self, teacher_logits: Dict[str, torch.Tensor]):
        return {k: self.adapters[k](v) for k, v in teacher_logits.items()}


class HDTSEConfidence(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp

    @torch.no_grad()
    def forward(self, student_logits, teacher_logits, teacher_order, targets):
        stacked = torch.stack([teacher_logits[k] for k in teacher_order], dim=1)  # (B,T,C)
        probs = F.softmax(stacked / self.temp, dim=-1)  # (B,T,C)

        # Hard labels: (B,)
        if targets.dim() == 1:
            idx = targets.to(dtype=torch.long, device=probs.device)
            conf = probs.gather(-1, idx[:, None, None]).squeeze(-1)  # (B,T)
            return normalize_lambdas(conf)

        # Soft labels (mixup/cutmix): (B,C)
        tgt = targets.to(dtype=probs.dtype, device=probs.device)
        conf = (probs * tgt[:, None, :]).sum(dim=-1)  # (B,T)
        return normalize_lambdas(conf)


class MultiTeacherDistillationLoss(nn.Module):
    def __init__(
        self,
        base_criterion,
        student_num_classes: int,
        teacher_names: List[str],
        distillation_type: str = "soft",
        alpha: float = 0.5,
        tau: float = 2.0,
        device=None,
        use_adapter: bool = True,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.teachers = FrozenTeacherEnsemble(teacher_names, self.device)
        self.adapter = TeacherLogitAdapter(self.teachers.teacher_order, student_num_classes).to(self.device) if use_adapter else None
        self.hdtse = HDTSEConfidence()

    def forward(self, inputs, outputs, targets):
        base_loss = self.base_criterion(outputs, targets)

        with torch.no_grad():
            t_logits = self.teachers(inputs)
        if self.adapter is not None:
            t_logits = self.adapter(t_logits)

        order = self.teachers.teacher_order
        lambdas = self.hdtse(outputs, t_logits, order, targets)
        fused = fuse_logits(t_logits, order, lambdas)

        kd = kd_soft(outputs, fused, self.tau) if self.distillation_type == "soft" else kd_hard(outputs, fused)
        return (1 - self.alpha) * base_loss + self.alpha * kd
