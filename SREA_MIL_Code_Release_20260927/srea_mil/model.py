"""SREA-MIL and paper ablation models.

The implementation follows the manuscript protocol:

* official DTFD-MIL first-tier modules (obtained separately from upstream);
* eight pseudo-bags and a 1280 -> 512 bias-free projection;
* eight learned evidence slots with 256 hidden dimensions;
* gated state aggregation and four inducing relation anchors;
* equal-weight branch-logit fusion;
* symmetric stop-gradient KL regularization during training;
* shared parameters for normal and suspected-abnormal feature streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dtfd_dependency import load_dtfd_layers


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    baseline_second_tier: bool = False
    evidence_tokenizer: bool = False
    gated_state_aggregation: bool = False
    inducing_relation_aggregation: bool = False
    mutual_consistency_regularization: bool = False


MODEL_SPECS: dict[str, ModelSpec] = {
    "dtfd_mil": ModelSpec("DTFD-MIL", baseline_second_tier=True),
    "tokenizer_mil": ModelSpec("Tokenizer-MIL", evidence_tokenizer=True),
    "state_mil": ModelSpec("State-MIL", gated_state_aggregation=True),
    "relation_mil": ModelSpec("Relation-MIL", inducing_relation_aggregation=True),
    "token_state_mil": ModelSpec(
        "Token-State-MIL", evidence_tokenizer=True, gated_state_aggregation=True
    ),
    "token_relation_mil": ModelSpec(
        "Token-Relation-MIL",
        evidence_tokenizer=True,
        inducing_relation_aggregation=True,
    ),
    "srea_mil_nomcr": ModelSpec(
        "SREA-MIL-NoMCR",
        evidence_tokenizer=True,
        gated_state_aggregation=True,
        inducing_relation_aggregation=True,
    ),
    "srea_mil": ModelSpec(
        "SREA-MIL",
        evidence_tokenizer=True,
        gated_state_aggregation=True,
        inducing_relation_aggregation=True,
        mutual_consistency_regularization=True,
    ),
}


@dataclass(frozen=True)
class SREAConfig:
    input_dim: int = 1280
    pseudo_dim: int = 512
    hidden_dim: int = 256
    num_groups: int = 8
    num_slots: int = 8
    num_anchors: int = 4
    num_classes: int = 2
    tokenizer_temperature: float = 0.25
    mutual_kl_weight: float = 0.05
    auxiliary_weight: float = 0.5
    eps: float = 1e-6


class DTFDFirstTier(nn.Module):
    """Source-faithful DTFD first tier and optional official second tier."""

    def __init__(self, cfg: SREAConfig, include_second_tier: bool) -> None:
        super().__init__()
        DimReduction, Classifier_1fc, Attention_Gated, Attention_with_Classifier = (
            load_dtfd_layers()
        )
        self.num_groups = cfg.num_groups
        self.dim_reduction = DimReduction(
            cfg.input_dim, cfg.pseudo_dim, numLayer_Res=0
        )
        self.attention = Attention_Gated(L=cfg.pseudo_dim, D=128, K=1)
        self.classifier = Classifier_1fc(cfg.pseudo_dim, cfg.num_classes, 0.25)
        self.second_tier = (
            Attention_with_Classifier(
                L=cfg.pseudo_dim,
                D=128,
                K=1,
                num_cls=cfg.num_classes,
                droprate=0.25,
            )
            if include_second_tier
            else None
        )

    def forward(
        self, features: Tensor, label: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must have shape [instances, channels]")
        groups = min(self.num_groups, features.shape[0])
        pseudo_features: list[Tensor] = []
        group_logits: list[Tensor] = []
        for chunk in torch.tensor_split(features, groups, dim=0):
            reduced = self.dim_reduction(chunk)
            attention = self.attention(reduced).squeeze(0)
            pooled = torch.einsum("nd,n->d", reduced, attention).unsqueeze(0)
            pseudo_features.append(pooled)
            group_logits.append(self.classifier(pooled))
        pseudo_bag = torch.cat(pseudo_features, dim=0)
        if self.training and label is not None:
            logits = torch.cat(group_logits, dim=0)
            targets = label.reshape(1).expand(logits.shape[0])
            group_ce = F.cross_entropy(logits, targets)
        else:
            group_ce = pseudo_bag.sum() * 0.0
        return pseudo_bag, group_ce


class EvidenceTokenizer(nn.Module):
    """Softly reorganize group descriptors into learned evidence slots."""

    def __init__(self, cfg: SREAConfig) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(cfg.pseudo_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.routing_score = nn.Linear(cfg.hidden_dim, 1)
        self.slot_polarity = nn.Parameter(torch.zeros(cfg.num_slots))
        self.queries = nn.Parameter(
            torch.randn(cfg.num_slots, cfg.hidden_dim) * 0.02
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(cfg.tokenizer_temperature).log()
        )
        self.eps = cfg.eps

    def forward(self, pseudo_bag: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if pseudo_bag.ndim != 2:
            raise ValueError("pseudo_bag must have shape [groups, channels]")
        projected = self.project(pseudo_bag)
        similarities = torch.einsum(
            "gd,kd->gk",
            F.normalize(projected, dim=-1),
            F.normalize(self.queries, dim=-1),
        )
        routing = self.routing_score(projected).squeeze(-1)
        similarities = similarities + routing[:, None] * self.slot_polarity[None, :]
        temperature = self.log_temperature.exp().clamp(0.05, 2.0)
        assignment = F.softmax(similarities / temperature, dim=-1)
        denominator = assignment.sum(dim=0).clamp_min(self.eps)
        slots = torch.einsum("gk,gd->kd", assignment, projected) / denominator[:, None]
        return slots, assignment, routing


class SharedEvidenceProjector(nn.Module):
    """Projection used by ablations that omit evidence tokenization."""

    def __init__(self, cfg: SREAConfig) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(cfg.pseudo_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )

    def forward(self, pseudo_bag: Tensor) -> Tensor:
        return self.project(pseudo_bag)


class StateEvidenceEncoder(nn.Module):
    """Short gated recurrence over the learned slot-index order."""

    def __init__(self, cfg: SREAConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.in_proj = nn.Linear(d, d)
        self.input_gate = nn.Linear(d, d)
        self.log_decay = nn.Parameter(torch.full((d,), -1.0))
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [slots, channels]")
        values = self.in_proj(tokens)
        decay = torch.sigmoid(self.log_decay)
        state = torch.zeros_like(values[0])
        outputs: list[Tensor] = []
        for current in values:
            gate = torch.sigmoid(self.input_gate(current))
            state = decay * state + (1.0 - decay) * gate * current
            outputs.append(state)
        sequence = self.norm(self.out_proj(torch.stack(outputs)) + tokens)
        return sequence, sequence.mean(dim=0, keepdim=True)


class InducingRelationEncoder(nn.Module):
    """Four-anchor relation exchange with O(Kr) affinity entries."""

    def __init__(self, cfg: SREAConfig) -> None:
        super().__init__()
        d, r = cfg.hidden_dim, cfg.num_anchors
        self.anchors = nn.Parameter(torch.randn(r, d) * 0.02)
        self.q = nn.Linear(d, r, bias=False)
        self.k = nn.Linear(d, r, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.anchor_update = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.scale = r ** -0.5

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [slots, channels]")
        token_to_anchor = F.softmax(
            self.q(tokens) @ self.k(self.anchors).T * self.scale, dim=-1
        )
        updated_anchors = token_to_anchor.T @ self.v(tokens)
        updated_anchors = updated_anchors + self.anchor_update(self.anchors)
        anchor_to_token = F.softmax(
            self.q(updated_anchors) @ self.k(tokens).T * self.scale, dim=-1
        )
        relation = anchor_to_token.T @ updated_anchors
        relation = self.norm(self.out(relation) + tokens)
        return relation, relation.mean(dim=0, keepdim=True)


def symmetric_stop_gradient_kl(left: Tensor, right: Tensor) -> Tensor:
    left_log = F.log_softmax(left, dim=-1)
    right_log = F.log_softmax(right, dim=-1)
    return 0.5 * (
        F.kl_div(left_log, right_log.exp().detach(), reduction="batchmean")
        + F.kl_div(right_log, left_log.exp().detach(), reduction="batchmean")
    )


class SREAStream(nn.Module):
    """One shared-parameter feature-stream core."""

    def __init__(
        self,
        model_name: str = "srea_mil",
        cfg: SREAConfig = SREAConfig(),
    ) -> None:
        super().__init__()
        if model_name not in MODEL_SPECS:
            raise ValueError(f"unknown model {model_name!r}: {sorted(MODEL_SPECS)}")
        self.model_name = model_name
        self.spec = MODEL_SPECS[model_name]
        self.cfg = cfg
        self.auxiliary_weight = cfg.auxiliary_weight
        self.first_tier = DTFDFirstTier(cfg, self.spec.baseline_second_tier)
        if self.spec.baseline_second_tier:
            return
        if self.spec.evidence_tokenizer:
            self.tokenizer = EvidenceTokenizer(cfg)
        else:
            self.projector = SharedEvidenceProjector(cfg)
        if model_name == "tokenizer_mil":
            self.token_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
            return
        if self.spec.gated_state_aggregation:
            self.state = StateEvidenceEncoder(cfg)
            self.state_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        if self.spec.inducing_relation_aggregation:
            self.relation = InducingRelationEncoder(cfg)
            self.relation_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward_stream(
        self, features: Tensor, label: Tensor | None = None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        pseudo_bag, group_ce = self.first_tier(features, label)
        diagnostics: dict[str, Tensor] = {
            "pseudo_bag": pseudo_bag,
            "group_ce": group_ce,
        }
        if self.spec.baseline_second_tier:
            assert self.first_tier.second_tier is not None
            logits = self.first_tier.second_tier(pseudo_bag).reshape(-1)
            return logits, group_ce, diagnostics

        if self.spec.evidence_tokenizer:
            tokens, assignment, routing = self.tokenizer(pseudo_bag)
            diagnostics.update(
                assignment=assignment,
                routing_score=routing,
                evidence_slots=tokens,
            )
        else:
            tokens = self.projector(pseudo_bag)
            diagnostics["aggregation_tokens"] = tokens

        if self.model_name == "tokenizer_mil":
            logits = self.token_head(tokens.mean(dim=0, keepdim=True)).reshape(-1)
            return logits, group_ce, diagnostics

        state_logits: Tensor | None = None
        relation_logits: Tensor | None = None
        if self.spec.gated_state_aggregation:
            state_tokens, state_repr = self.state(tokens)
            state_logits = self.state_head(state_repr)
            diagnostics.update(state_tokens=state_tokens, state_logits=state_logits)
        if self.spec.inducing_relation_aggregation:
            relation_tokens, relation_repr = self.relation(tokens)
            relation_logits = self.relation_head(relation_repr)
            diagnostics.update(
                relation_tokens=relation_tokens, relation_logits=relation_logits
            )

        if state_logits is not None and relation_logits is not None:
            logits = 0.5 * (state_logits + relation_logits)
        elif state_logits is not None:
            logits = state_logits
        elif relation_logits is not None:
            logits = relation_logits
        else:
            raise RuntimeError("the selected model has no aggregation branch")

        consistency = logits.sum() * 0.0
        if self.spec.mutual_consistency_regularization:
            assert state_logits is not None and relation_logits is not None
            consistency = symmetric_stop_gradient_kl(state_logits, relation_logits)
        diagnostics["mutual_consistency_loss"] = consistency
        auxiliary = group_ce + self.cfg.mutual_kl_weight * consistency
        return logits.reshape(-1), auxiliary, diagnostics


class TwoStreamSREAMIL(nn.Module):
    """Process two candidate streams with one shared SREA core."""

    def __init__(
        self,
        model_name: str = "srea_mil",
        cfg: SREAConfig = SREAConfig(),
    ) -> None:
        super().__init__()
        self.core = SREAStream(model_name, cfg)
        self.auxiliary_weight = cfg.auxiliary_weight

    def forward(
        self,
        abnormal: Tensor,
        abnormal_mask: Tensor,
        normal: Tensor,
        normal_mask: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_logits: list[Tensor] = []
        batch_auxiliary: list[Tensor] = []
        for index in range(abnormal.shape[0]):
            logits_per_stream: list[Tensor] = []
            auxiliary_per_stream: list[Tensor] = []
            label = labels[index] if labels is not None else None
            if bool(abnormal_mask[index].any()):
                logits, auxiliary, _ = self.core.forward_stream(
                    abnormal[index][abnormal_mask[index]], label
                )
                logits_per_stream.append(logits)
                auxiliary_per_stream.append(auxiliary)
            if bool(normal_mask[index].any()):
                logits, auxiliary, _ = self.core.forward_stream(
                    normal[index][normal_mask[index]], label
                )
                logits_per_stream.append(logits)
                auxiliary_per_stream.append(auxiliary)
            if not logits_per_stream:
                raise RuntimeError("a slide has no valid feature stream")
            batch_logits.append(torch.stack(logits_per_stream).mean(dim=0))
            batch_auxiliary.append(torch.stack(auxiliary_per_stream).mean())
        logits = torch.stack(batch_logits)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "auxiliary_loss": torch.stack(batch_auxiliary).mean(),
        }


def build_model(
    model_name: str = "srea_mil", cfg: SREAConfig = SREAConfig()
) -> TwoStreamSREAMIL:
    return TwoStreamSREAMIL(model_name, cfg)
