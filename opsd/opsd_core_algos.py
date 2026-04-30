# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2020-2025 The HuggingFace Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Core algorithms for the OPSD recipe.

Implements the generalized Jensen-Shannon divergence loss used by the
On-Policy Self-Distillation update. The conventions (sign, masking,
top-k restriction, per-token clipping) follow the TRL GKD reference
trainer so that token-level divergences are byte-identical given the
same logits, see https://huggingface.co/papers/2306.13649.
"""

import torch
import torch.nn.functional as F


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    logits_are_probs: bool = False,
    top_k: int | None = None,
    token_clip: float | None = None,
) -> torch.Tensor:
    """Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation.

    See Eq. (1) of https://huggingface.co/papers/2306.13649 for the canonical
    definition. PyTorch's ``F.kl_div`` flips the conventional argument order; we
    follow the same sign convention as the upstream TRL OPSD trainer so that
    ``beta=0`` matches forward-KL ``KL(teacher || student)``.

    Args:
        student_logits: Tensor of shape ``(batch_size, sequence_length, vocab_size)``.
            Gradients flow through this tensor.
        teacher_logits: Tensor of shape ``(batch_size, sequence_length, vocab_size)``.
            Detached: the teacher is treated as a frozen target distribution.
        labels: Optional tensor of shape ``(batch_size, sequence_length)`` with
            ``-100`` at positions to ignore (typically the prompt tokens and any
            padding).
        beta: Mixture coefficient between student and teacher. ``beta=0`` is
            forward-KL, ``beta=1`` is reverse-KL, ``beta=0.5`` is symmetric JSD.
        temperature: Softmax temperature applied to both logits before
            computing log-probabilities.
        reduction: ``batchmean`` (default), ``sum``, ``mean``, or ``none``.
            ``batchmean`` divides by the number of unmasked tokens.
        logits_are_probs: If ``True``, treat the inputs as already-normalized
            probabilities and skip softmax. Used by callers that want to inject
            externally renormalized distributions.
        top_k: If set, restrict the loss to the top-k tokens of the teacher
            distribution. Both distributions are renormalized over the selected
            tokens before the divergence is computed.
        token_clip: If set, clip per-token divergence values to this maximum
            before reduction. Stops a few high-divergence "style" tokens from
            dominating the gradient signal over math tokens.

    Returns:
        Scalar tensor with the JSD loss (or per-token tensor when
        ``reduction='none'``).
    """
    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        # log((1 - beta) * p_student + beta * p_teacher) computed in log-space
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_tensor),
                    teacher_log_probs + torch.log(beta_tensor),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student

    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    mask = None
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    if reduction == "batchmean":
        if mask is not None:
            denom = mask.sum().clamp_min(1)
            return jsd.sum() / denom
        return jsd.sum() / jsd.size(0)
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd
