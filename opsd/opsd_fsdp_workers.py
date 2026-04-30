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
"""FSDP workers for the OPSD recipe.

We subclass ``ActorRolloutRefWorker`` so a single Ray actor hosts the student
(actor + rollout) **and** the frozen teacher (ref). Both forwards happen
locally inside ``update_actor_opsd``, so the teacher's full vocabulary logits
never need to cross the Ray boundary.
"""

import logging
import os

import psutil
import torch
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class OPSDActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Hybrid actor+rollout+ref worker for OPSD.

    Inherits the entire model/optimizer/rollout build path from
    ``AsyncActorRolloutRefWorker`` — verl's FSDP worker that exposes the async
    rollout interface (``get_zeromq_address``, ``wake_up``/``sleep``,
    chat_completion endpoints) the vLLM HTTP server speaks. Subclassing the
    sync ``ActorRolloutRefWorker`` instead leaves the worker without those
    methods and the `vLLMHttpServer` startup blows up at
    ``worker.get_zeromq_address.remote()``. We only override the ``self.actor``
    instantiation so we get an ``OPSDDataParallelPPOActor`` with a handle to
    the colocated teacher ``self.ref_module_fsdp``.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):  # type: ignore[override]
        # Build the actor (student) module + optimizer, the rollout, and the
        # reference (teacher) module via the parent. After this call we have
        # ``self.actor_module_fsdp`` (student, has grad), ``self.ref_module_fsdp``
        # (teacher, frozen), ``self.actor`` (default DataParallelPPOActor),
        # ``self.ref_policy``, and the rollout engine.
        super().init_model()

        if not (self._is_actor and self._is_ref):
            # Standalone actor or ref worker: nothing OPSD-specific to wire up.
            # All update_actor_opsd calls assume the colocated layout.
            return

        # Replace the default actor with an OPSD-aware actor that owns a handle
        # to the teacher FSDP module so it can run both forwards in one step.
        from recipe.opsd.opsd_dp_actor import OPSDDataParallelPPOActor

        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = OPSDDataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
            teacher_module=self.ref_module_fsdp,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update_opsd")
    def update_actor_opsd(self, data: DataProto):
        """OPSD update step: dispatched per DP rank, runs JSD locally.

        Mirrors ``ActorRolloutRefWorker.update_actor`` (FSDP-state management,
        timing, MFU bookkeeping) but invokes ``update_policy_opsd`` instead of
        the PPO ``update_policy``.
        """
        assert self._is_actor, "OPSD update_actor_opsd called on non-actor worker"
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            # Teacher must also be GPU-resident for the JSD forward pass.
            load_fsdp_model_to_gpu(self.ref_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # micro-batches will move to GPU inside update_policy_opsd

            with Timer(name="update_policy_opsd", logger=None) as timer:
                metrics = self.actor.update_policy_opsd(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # OPSD adds a teacher forward (no grad, ~2 * params * tokens) on top
            # of the student fwd+bwd (~6 * params * tokens), so the per-epoch
            # work is 4/3 of PPO's; scale ppo_epochs accordingly.
            metrics["perf/mfu/actor"] = (
                estimated_flops * (4.0 / 3.0) * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        log_gpu_memory_usage("After update_actor_opsd", logger=logger)

        return output
