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
"""Lightweight config validation for the OPSD recipe.

OPSD has no critic and no reward-model worker, so the standard
``recipe.spin.utils.validate_config`` checks for those components are
intentionally dropped here. Otherwise the pre-flight contract is the same.
"""

from omegaconf import DictConfig, OmegaConf


def validate_opsd_config(config: DictConfig, use_reference_policy: bool) -> None:
    """Validate an OPSD config before workers are spun up.

    Args:
        config: full OmegaConf DictConfig built from ``opsd_trainer.yaml``
            plus command-line overrides.
        use_reference_policy: forwarded from ``need_reference_policy`` in
            ``main_opsd``. Always ``True`` for OPSD because the teacher is
            wired through ``Role.ActorRolloutRef``.
    """
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    assert real_train_batch_size % n_gpus == 0, (
        f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
    )

    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str, param: str):
        param_per_gpu = f"{param}_per_gpu"
        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")
        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported "
                f"(the former is deprecated)."
            )

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        check_mutually_exclusive(
            config.actor_rollout_ref.actor.ppo_micro_batch_size,
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
            "actor_rollout_ref.actor",
            "ppo_micro_batch_size",
        )
        if use_reference_policy:
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
                "log_prob_micro_batch_size",
            )
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
            "log_prob_micro_batch_size",
        )

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
        sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
            assert (
                config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size
                == 0
            )
            assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        if (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

    # OPSD-specific: ``OPSDDataParallelPPOActor._forward_logits`` always runs
    # a padded forward and slices ``logits[:, -response_length-1:-1, :]``. The
    # remove-padding fast path in the parent ``DataParallelPPOActor`` would
    # require unpacking and repadding full-vocab logits per micro-batch, which
    # is not implemented and would explode peak memory on Qwen3.5's 248K
    # vocab. Reject it up-front so misconfiguration cannot silently produce
    # wrong logits.
    if config.actor_rollout_ref.model.get("use_remove_padding", False):
        raise ValueError(
            "OPSD does not support actor_rollout_ref.model.use_remove_padding=True; "
            "the JSD forward materialises full-vocab logits and only supports "
            "padded inputs. Set use_remove_padding=False (also disables "
            "ulysses_sequence_parallel_size>1)."
        )

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines "
            "as a whole batch, which will schedule the memory themselves."
        )

    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # OPSD-specific: the JSD loss requires either beta in [0,1] (forward-KL,
    # JSD, reverse-KL) or beta=0/1 (KL only). reject anything else now so we
    # don't crash 30 minutes into a run.
    opsd_cfg = OmegaConf.select(config.algorithm, "opsd", default=None)
    assert opsd_cfg is not None, "config.algorithm.opsd block is missing."
    beta = float(opsd_cfg.get("beta", 0.0))
    assert 0.0 <= beta <= 1.0, f"algorithm.opsd.beta must be in [0, 1], got {beta}."

    print("[validate_opsd_config] All configuration checks passed successfully!")
