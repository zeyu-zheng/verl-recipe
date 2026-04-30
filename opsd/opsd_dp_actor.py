# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""OPSD-specific ``DataParallelPPOActor``.

The student forward pass needs to expose **full vocabulary logits** so we can
compute the generalized Jensen-Shannon divergence against the teacher; verl's
default actor only emits ``log_probs`` of the sampled tokens. We subclass the
default actor and add:

* ``_forward_logits``: a thin micro-batch forward pass that returns
  ``(bsz, response_length, vocab_size)`` logits aligned to the response slice.
* ``update_policy_opsd``: an OPSD-specific update step that runs the student
  (with gradient) and the frozen teacher (under ``torch.no_grad``) on their
  respective inputs and computes JSD.

The teacher module is held by ``OPSDRolloutRefWorker`` (the FSDP'd ref clone)
and passed in via the ``teacher_module`` kwarg, so both forward passes happen
on the same Ray actor and the teacher logits never cross the worker boundary.
"""

import logging
import os
from collections import defaultdict

import torch
from recipe.opsd.opsd_core_algos import generalized_jsd_loss
from torch import nn

from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

__all__ = ["OPSDDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OPSDDataParallelPPOActor(DataParallelPPOActor):
    """Student-side actor for On-Policy Self-Distillation.

    Holds a reference to the frozen teacher module so it can run both forward
    passes inside a single ``update_policy_opsd`` call.
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        teacher_module: nn.Module,
    ):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        # Teacher is the colocated reference FSDP module that is never updated
        # (ref.update_freq == -1 in the YAML). It still lives on the same Ray
        # actor as the student, so we can run both forwards back-to-back without
        # serializing logits across the Ray boundary.
        assert teacher_module is not None, "OPSD requires a teacher module on the actor worker."
        self.teacher_module = teacher_module

    def _forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
        module: nn.Module,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Forward a single micro-batch and return logits for the response slice.

        Returns a tensor of shape ``(bsz, response_length, vocab_size)`` aligned
        such that position ``t`` predicts response token ``t``. Mirrors the
        non-rmpad branch of ``DataParallelPPOActor._forward_micro_batch`` so the
        slicing convention is identical to the rest of verl.

        ``temperature`` is applied via ``logits / temperature`` *before* the
        return, matching the way ``generalized_jsd_loss`` further re-divides by
        its own temperature when computing log-softmax. This dual scaling is
        intentional to stay byte-identical with the TRL reference (which also
        passes raw logits to ``F.log_softmax`` after dividing by temperature).
        """
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        logits = output.logits
        if temperature != 1.0:
            logits = logits / temperature
        # Predict response token t with the logits at position prompt_len + t - 1,
        # i.e. slice [-response_length-1 : -1] of the full sequence. This matches
        # `DataParallelPPOActor._forward_micro_batch` and `OPSDTrainer.compute_loss`.
        return logits[:, -response_length - 1 : -1, :]

    @GPUMemoryLogger(role="opsd actor", logger=logger)
    def update_policy_opsd(self, data: DataProto):
        """OPSD update step.

        Expects the following keys in ``data.batch``:

        * ``input_ids`` / ``attention_mask`` / ``position_ids`` – student inputs
          (prompt + generated response, right-padded to the same length per
          micro-batch).
        * ``responses`` – the generated response slice ``(bsz, response_length)``.
        * ``response_mask`` – ``(bsz, response_length)`` 0/1 mask over response
          tokens.
        * ``teacher_input_ids`` / ``teacher_attention_mask`` /
          ``teacher_position_ids`` – teacher inputs (teacher prompt + the same
          generated response).
        * ``teacher_response_offset`` – ``(bsz,)`` int tensor giving the index
          in the teacher sequence where the response slice ends. Equivalent to
          ``teacher_prompt_length + response_length`` per example. Provided so
          the trainer can pad teacher prompts to varying lengths without losing
          alignment.

        And in ``meta_info``:

        * ``opsd``: dict carrying the JSD hyperparameters (``beta``,
          ``temperature``, ``forward_temperature``, ``token_clip``, ``top_k``)
          dehydrated from ``algorithm.opsd`` by the trainer.
        """
        self.actor_module.train()
        self.teacher_module.eval()
        for p in self.teacher_module.parameters():
            p.requires_grad_(False)

        # ``forward_temperature`` is applied to the raw forward-pass logits;
        # ``temperature`` is applied inside generalized_jsd_loss. The TRL reference
        # only divides inside the loss (forward_temperature == 1.0).
        opsd_cfg = data.meta_info["opsd"]
        forward_temperature = float(opsd_cfg["forward_temperature"])
        loss_temperature = float(opsd_cfg["temperature"])
        beta = float(opsd_cfg["beta"])
        token_clip = opsd_cfg["token_clip"]
        token_clip = float(token_clip) if token_clip is not None else None
        top_k = opsd_cfg["top_k"]
        top_k = int(top_k) if top_k else None

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
            "teacher_response_offset",
        ]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=[])

        # Stash the real student attention mask before dynamic batching
        # overwrites ``attention_mask`` with the synthetic packing mask
        # (see the loop body below).
        data.batch["student_attention_mask"] = data.batch["attention_mask"].clone()

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics: dict[str, list[float]] = defaultdict(list)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    # ``rearrange_micro_batches`` packs by ``attention_mask.sum(-1)``,
                    # so we feed it a synthetic mask whose per-row length is
                    # ``max(student_len, teacher_len)``. The teacher's NL prefix
                    # is typically 5-10x longer than the student's prompt, so
                    # packing on the student mask alone would under-estimate
                    # peak memory (full-vocab logits are materialised for both
                    # forwards). The real student mask is preserved as
                    # ``student_attention_mask`` for the actual forward.
                    pack_lens = torch.maximum(
                        mini_batch.batch["student_attention_mask"].sum(dim=-1),
                        mini_batch.batch["teacher_attention_mask"].sum(dim=-1),
                    )
                    max_pack_len = int(pack_lens.max().item())
                    pack_mask = (
                        torch.arange(max_pack_len, device=pack_lens.device).unsqueeze(0) < pack_lens.unsqueeze(-1)
                    ).to(mini_batch.batch["student_attention_mask"].dtype)
                    mini_batch.batch["attention_mask"] = pack_mask

                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # Total response token count across the entire mini-batch. We
                # weight each micro-batch loss by (n_tokens_micro / n_tokens_mini)
                # so that gradient accumulation reproduces a single-pass
                # batchmean reduction even when dynamic batching gives
                # heterogeneous micro-batches.
                total_response_tokens = max(int(mini_batch.batch["response_mask"].sum().item()), 1)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_length = inputs["responses"].size(-1)
                    response_mask = inputs["response_mask"]
                    n_tokens_micro = int(response_mask.sum().item())
                    if n_tokens_micro == 0:
                        # All-pad micro-batch: skip to avoid divide-by-zero.
                        continue

                    # ``inputs["attention_mask"]`` is the synthetic packing mask
                    # under dynamic batching; ``student_attention_mask`` is the
                    # real per-token student mask aligned to ``input_ids``. Use
                    # the latter for the student forward.
                    student_attention_mask = inputs.get("student_attention_mask", inputs["attention_mask"])
                    student_logits = self._forward_logits(
                        input_ids=inputs["input_ids"],
                        attention_mask=student_attention_mask,
                        position_ids=inputs["position_ids"],
                        response_length=response_length,
                        module=self.actor_module,
                        temperature=forward_temperature,
                    )

                    with torch.no_grad():
                        teacher_logits = self._forward_logits(
                            input_ids=inputs["teacher_input_ids"],
                            attention_mask=inputs["teacher_attention_mask"],
                            position_ids=inputs["teacher_position_ids"],
                            response_length=response_length,
                            module=self.teacher_module,
                            temperature=forward_temperature,
                        )

                    # Mask: -100 wherever response_mask == 0 so JSD ignores
                    # padding tokens (and any future padded-to-equal-length
                    # truncations). We mirror TRL's labels==-100 convention.
                    labels = torch.where(
                        response_mask.bool(),
                        torch.zeros_like(response_mask, dtype=torch.long),
                        torch.full_like(response_mask, -100, dtype=torch.long),
                    )

                    loss = generalized_jsd_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        labels=labels,
                        beta=beta,
                        temperature=loss_temperature,
                        token_clip=token_clip,
                        top_k=top_k,
                        reduction="batchmean",
                    )

                    # ``loss`` is already a per-token mean over n_tokens_micro;
                    # multiplying by (n_tokens_micro / total_response_tokens)
                    # turns it into "this micro-batch's token-weighted share of
                    # the mini-batch mean" so that summing across micro-batches
                    # reproduces a single-pass batchmean.
                    loss_weight = n_tokens_micro / total_response_tokens
                    scaled_loss = loss * loss_weight
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    metrics["actor/jsd_loss"].append(loss.detach().item())
                    metrics["actor/jsd_loss_finite"].append(float(torch.isfinite(loss).item()))

                grad_norm = self._optimizer_step()
                metrics["actor/grad_norm"].append(grad_norm.detach().item())

        self.actor_optimizer.zero_grad()

        # Average per-call metrics for compatibility with verl's reduce_metrics.
        out_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if len(v) == 0:
                continue
            out_metrics[k] = float(sum(v) / len(v))
        return out_metrics
