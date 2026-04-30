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
"""On-Policy Self-Distillation Ray trainer.

Inherits dataloader, validation, checkpointing and worker init from
``RayPPOTrainer``; overrides ``fit`` to replace PPO's advantage / critic /
KL machinery with a single OPSD step:

1. Generate student rollouts (vLLM via the colocated rollout engine).
2. Build teacher inputs by tokenizing the per-row ``teacher_raw_prompt``
   (a chat-messages list carrying any extra context the teacher should
   condition on) and concatenating with the generated response. Rows
   without a ``teacher_raw_prompt`` fall back to a clone of the student
   inputs (degenerate teacher, JSD == 0).
3. Call ``actor_rollout_wg.update_actor_opsd``: the worker computes student
   and teacher logits locally and backprops the JSD.

The reward function (a user-supplied ``custom_reward_function``) is wired
through purely for validation and rollout dumping; the training loss
itself is the JSD and never consumes the reward.
"""

import uuid
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import marked_timer
from verl.utils.tracking import Tracking


def _build_teacher_tensors(
    batch: DataProto,
    tokenizer,
    response_length: int,
    max_teacher_prompt_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Tokenize ``teacher_raw_prompt`` (chat messages list) and concat with the
    student-generated response to form teacher input tensors.

    Rows that carry no ``teacher_raw_prompt`` fall back to the student inputs,
    yielding a JSD of zero and still exercising the full pipeline.

    Args:
        batch: full DataProto post-rollout. Must contain ``input_ids``,
            ``attention_mask``, ``position_ids``, ``responses`` and either
            ``teacher_raw_prompt`` (object array of chat-message lists) or
            no teacher prompt at all.
        tokenizer: HF tokenizer for applying the chat template to the teacher
            messages.
        response_length: max response length per example (the response slice
            is right-padded to this size).
        max_teacher_prompt_length: cap for teacher prompt tokens; longer
            prompts are right-truncated.
        pad_token_id: tokenizer pad id used for both prompt and response pad.

    Returns:
        dict with ``teacher_input_ids``, ``teacher_attention_mask``,
        ``teacher_position_ids`` (all on the same device as ``batch.batch``)
        and ``teacher_response_offset`` (per-example end index of response).
    """
    student_input_ids: torch.Tensor = batch.batch["input_ids"]
    bsz = student_input_ids.shape[0]
    device = student_input_ids.device

    teacher_raw_prompt = batch.non_tensor_batch.get("teacher_raw_prompt", None)
    has_teacher_prompt = (
        teacher_raw_prompt is not None
        and len(teacher_raw_prompt) == bsz
        and all(teacher_raw_prompt[i] is not None for i in range(bsz))
    )

    if not has_teacher_prompt:
        # Degenerate fallback: teacher == student. JSD is identically zero.
        return {
            "teacher_input_ids": student_input_ids.clone(),
            "teacher_attention_mask": batch.batch["attention_mask"].clone(),
            "teacher_position_ids": batch.batch["position_ids"].clone(),
            "teacher_response_offset": batch.batch["attention_mask"].sum(dim=-1).long(),
        }

    responses: torch.Tensor = batch.batch["responses"]  # (bsz, response_length)

    teacher_prompt_ids_list: list[list[int]] = []
    for i in range(bsz):
        messages = list(teacher_raw_prompt[i])
        teacher_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tokenizer(
            teacher_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_teacher_prompt_length,
        )["input_ids"]
        teacher_prompt_ids_list.append(ids)

    teacher_prompt_lens = [len(ids) for ids in teacher_prompt_ids_list]
    max_teacher_prompt_len = max(teacher_prompt_lens)

    # Left-pad the teacher prompt so the response slice is at the end of the
    # sequence: this matches verl's right-side response convention and lets us
    # reuse the same `[-response_length-1:-1]` logits slice as the student.
    padded_prompt_ids = torch.full((bsz, max_teacher_prompt_len), pad_token_id, dtype=responses.dtype, device=device)
    prompt_attn = torch.zeros((bsz, max_teacher_prompt_len), dtype=torch.long, device=device)
    for i, ids in enumerate(teacher_prompt_ids_list):
        n = len(ids)
        padded_prompt_ids[i, max_teacher_prompt_len - n :] = torch.tensor(ids, dtype=responses.dtype, device=device)
        prompt_attn[i, max_teacher_prompt_len - n :] = 1

    # Response attention mask is the same shape as the student's response mask
    # because we're feeding the same generated tokens through the teacher.
    student_response_mask = batch.batch["attention_mask"][:, -response_length:]
    teacher_input_ids = torch.cat([padded_prompt_ids, responses], dim=1)
    teacher_attention_mask = torch.cat([prompt_attn, student_response_mask], dim=1)
    # Position ids: cumsum-style ids that are 0 on padding and 0..L-1 over the
    # unpadded prefix, matching verl's compute_position_id_with_mask helper.
    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

    teacher_response_offset = torch.tensor(
        [t_len + int(student_response_mask[i].sum().item()) for i, t_len in enumerate(teacher_prompt_lens)],
        dtype=torch.long,
        device=device,
    )

    return {
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attention_mask,
        "teacher_position_ids": teacher_position_ids,
        "teacher_response_offset": teacher_response_offset,
    }


class RayOPSDTrainer(RayPPOTrainer):
    """Ray-based driver for OPSD distillation.

    Reuses every helper on ``RayPPOTrainer`` except the training loop, which
    is replaced with an OPSD-specific ``fit``. The reward function is kept
    around solely for validation and optional rollout dumping; training
    updates are driven by the JSD loss inside the actor worker.

    The colocated teacher is exposed via ``Role.ActorRolloutRef`` (so
    ``use_reference_policy=True``) but its forward is consumed inside
    ``update_actor_opsd`` rather than ``compute_ref_log_prob``. The critic
    is disabled at the YAML level (``critic.enable: False`` in
    ``opsd_trainer.yaml``) so ``need_critic`` returns False without
    touching ``algorithm.adv_estimator``.
    """

    def fit(self):  # type: ignore[override]
        """OPSD training loop.

        Per step:
          1. Read a batch.
          2. Pop generation inputs, repeat by rollout.n, and generate via the
             async rollout manager (vLLM).
          3. Union rollouts back into the batch and compute response_mask /
             global_token_num for FLOPs accounting.
          4. Build teacher inputs (tokenized ``teacher_raw_prompt`` +
             response, or a student-clone fallback when no teacher prompt
             is available).
          5. Call ``update_actor_opsd`` to compute JSD and step the optimizer.
          6. Optionally run validation and checkpoint.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        # Optional validation before training. The custom reward function
        # (e.g. a verifier) is the canonical metric for OPSD success even
        # though it's not part of the training loss.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPSD Training Progress")
        self.global_steps += 1
        last_val_metrics: Optional[dict] = None
        max_teacher_prompt_length = int(OmegaConf.select(self.config.data, "max_teacher_prompt_length", default=8192))

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict = {}
                timing_raw: dict = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Async rollout drops every non-tensor key except __num_turns__/extra_fields.
                # Stash the OPSD-specific teacher prompt and re-attach it after rollout so
                # we can tokenize it for the actor's teacher forward pass.
                stashed_teacher_raw_prompt = batch.non_tensor_batch.pop("teacher_raw_prompt", None)

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # Repeat the prompt-side batch to align with rollout.n responses.
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if stashed_teacher_raw_prompt is not None:
                        repeated_teacher = np.repeat(
                            stashed_teacher_raw_prompt,
                            self.config.actor_rollout_ref.rollout.n,
                            axis=0,
                        )
                        batch.non_tensor_batch["teacher_raw_prompt"] = repeated_teacher

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    response_length = batch.batch["responses"].size(-1)
                    with marked_timer("teacher_inputs", timing_raw, color="cyan"):
                        teacher_tensors = _build_teacher_tensors(
                            batch=batch,
                            tokenizer=self.tokenizer,
                            response_length=response_length,
                            max_teacher_prompt_length=max_teacher_prompt_length,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        for k, v in teacher_tensors.items():
                            batch.batch[k] = v

                    # Pass OPSD-specific hyperparameters through to the actor.
                    batch.meta_info["opsd"] = OmegaConf.to_container(self.config.algorithm.opsd, resolve=True)

                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self.actor_rollout_wg.update_actor_opsd(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # Optional rollout dump. ``_log_rollout_data`` requires
                    # ``token_level_scores``; we run the user's custom reward
                    # function here purely for the dump (training itself
                    # does not consume the reward).
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir and self.reward_fn is not None:
                        with marked_timer("reward_for_dump", timing_raw, color="yellow"):
                            reward_out = self.reward_fn(batch, return_dict=True)
                            if isinstance(reward_out, dict):
                                batch.batch["token_level_scores"] = reward_out["reward_tensor"]
                                reward_extra_infos = reward_out.get("reward_extra_info", {}) or {}
                            else:
                                batch.batch["token_level_scores"] = reward_out
                                reward_extra_infos = {}
                        self._log_rollout_data(batch, reward_extra_infos, timing_raw, rollout_data_dir)

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
