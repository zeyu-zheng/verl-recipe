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
"""Hydra entry point for the OPSD recipe.

Usage::

    python -m recipe.opsd.main_opsd \\
        --config-path=$(pwd)/recipe/opsd/config \\
        --config-name=opsd_trainer \\
        actor_rollout_ref.model.path=<model> \\
        custom_reward_function.path=<path/to/reward.py> \\
        ...

The TaskRunner is launched as a Ray actor so the driver process is not
scheduled on the head node (matches the verl PPO entry-point convention).
"""

import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.trainer.ppo.utils import need_reference_policy


@hydra.main(config_path="config", config_name="opsd_trainer", version_base=None)
def main(config):
    run_opsd(config)


def run_opsd(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("OPSD recipe currently only supports FSDP/FSDP2 strategies.")

        from recipe.opsd.opsd_fsdp_workers import OPSDActorRolloutRefWorker
        from recipe.opsd.opsd_ray_trainer import RayOPSDTrainer

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        from verl.trainer.ppo.utils import Role

        role_worker_mapping: dict[Role, ray.actor.ActorClass] = {
            Role.ActorRolloutRef: ray.remote(OPSDActorRolloutRefWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRolloutRef: global_pool_id}

        # OPSD has no critic and no reward model; the user's custom reward
        # function is the only "reward function" used (validation only).
        from recipe.opsd.opsd_utils import validate_opsd_config

        validate_opsd_config(
            config=config,
            use_reference_policy=need_reference_policy(role_worker_mapping),
        )

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        from verl.workers.reward_manager import get_reward_manager_cls

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)

        compute_score = get_custom_reward_fn(config)
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        # Training does not consume the reward; we still build one for symmetry
        # with verl's trainer plumbing and to enable rollout-reward dumping.
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayOPSDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
