# Recipe: On-Policy Self-Distillation (OPSD)

## Required `verl` version

See [`./REQUIRED_VERL.txt`](./REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.

## Overview

On-Policy Self-Distillation (OPSD) is a knowledge-distillation-style RL recipe in which a single base model plays both **student** (rolls out completions, receives gradient) and **teacher** (frozen, conditions on additional context, supplies the target distribution). Each step:

1. The student rolls out responses for a batch of prompts under the standard verl rollout engine (vLLM).
2. The teacher copy of the same model — wired in via `Role.ActorRolloutRef` and held in FSDP on the same Ray actor — runs a no-grad forward over the **per-row teacher prompt** (`teacher_raw_prompt`) concatenated with the student's response.
3. The actor backprops a generalized Jensen-Shannon divergence between the student's and teacher's full-vocabulary logits over the response slice.

There is no critic, no reward model, and no advantage / KL term — the loss is purely the JSD. A `custom_reward_function` is still wired through verl's standard hooks, but it is consumed only for validation metrics and (optionally) rollout dumping.

The colocated layout — student and teacher live in the same Ray actor — means the teacher's full-vocab logits never cross the worker boundary, which is otherwise prohibitive for large vocabularies (e.g. ~248K on Qwen3.5).

## Dataset contract

Each row in the parquet train / val files must carry the standard verl fields (`prompt`, `data_source`, `extra_info`, `reward_model`, ...). To activate the teacher branch, also include a `teacher_raw_prompt` column whose value is a chat-messages list (the same shape `tokenizer.apply_chat_template` consumes), e.g.:

```python
example["teacher_raw_prompt"] = [
    {"role": "user", "content": "Here is some extra context the teacher gets to see..."},
    {"role": "assistant", "content": "...some scaffolding response."},
    {"role": "user", "content": example["prompt"][0]["content"]},
]
```

Rows with `teacher_raw_prompt is None` (or missing) fall back to a clone of the student inputs, yielding a JSD of zero — useful for ablating the teacher signal on a per-row basis.

## Quickstart

1. Convert your dataset to parquet with the layout above; place at `${RAY_DATA_HOME}/data/your_dataset.parquet`.
2. Provide a `custom_reward_function` (e.g. `verl/utils/reward_score/<your_task>.py::compute_score`) for validation only.
3. Submit:

```bash
cd verl  # repo root
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265"
export WORKING_DIR="${PWD}"
export RUNTIME_ENV="${WORKING_DIR}/recipe/opsd/runtime_env.yaml"

python -m recipe.opsd.main_opsd \
    --config-path=$(pwd)/recipe/opsd/config \
    --config-name=opsd_trainer \
    actor_rollout_ref.model.path=<HF_MODEL_OR_LOCAL_PATH> \
    data.train_files=${RAY_DATA_HOME}/data/your_dataset.parquet \
    data.val_files=${RAY_DATA_HOME}/data/your_val.parquet \
    custom_reward_function.path=<path/to/reward.py> \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

## Configuration

The recipe extends `ppo_trainer` (see `defaults` in `config/opsd_trainer.yaml`); only the OPSD-specific knobs are documented here.

### `algorithm.opsd.*`

```yaml
algorithm:
  opsd:
    beta: 0.0              # 0 -> forward-KL(teacher||student), 0.5 -> symmetric JSD, 1.0 -> reverse-KL
    temperature: 1.0       # softmax temperature inside the JSD log-softmax
    forward_temperature: 1.0  # pre-softmax temperature applied to raw forward-pass logits (TRL parity = 1.0)
    token_clip: null       # per-token JSD ceiling; null disables. Recommended to suppress a few high-divergence "style" tokens.
    top_k: null            # restrict JSD to teacher's top-k tokens; null disables.
```

The JSD implementation in [`opsd_core_algos.py`](./opsd_core_algos.py) follows the conventions of the TRL GKD trainer (see Eq. (1) of the [GKD paper](https://huggingface.co/papers/2306.13649)) so that token-level divergences are byte-identical given the same logits.

### `data.max_teacher_prompt_length`

```yaml
data:
  max_teacher_prompt_length: 8192  # cap on the tokenized teacher_raw_prompt; longer prompts are right-truncated
```

The teacher input layout is `[teacher_prompt | response]`, so the actor's per-GPU dynamic-batching budget must fit `max_teacher_prompt_length + max_response_length`.

### Hard constraints

- `actor_rollout_ref.model.use_remove_padding` **must be `False`**. The student-side forward in [`opsd_dp_actor.py`](./opsd_dp_actor.py) materialises full-vocabulary logits and only supports padded inputs; the recipe rejects `use_remove_padding=True` in [`opsd_utils.validate_opsd_config`](./opsd_utils.py).
- `critic.enable: False` and `reward_model.enable: False` (already set by `opsd_trainer.yaml`).
- Use `algorithm.opsd.beta in [0, 1]` (also enforced by the validator).

## How the loss is computed

```python
loss = generalized_jsd_loss(
    student_logits,             # (B, T_resp, V) — gradient flows through this
    teacher_logits.detach(),    # (B, T_resp, V) — frozen teacher
    labels=labels,              # -100 on prompt / pad, 0 on response tokens
    beta=opsd.beta,
    temperature=opsd.temperature,
    token_clip=opsd.token_clip,
    top_k=opsd.top_k,
    reduction="batchmean",      # divides by number of unmasked tokens
)
```

Per-token losses are computed over the full vocabulary (or the teacher's top-k slice when `top_k` is set), masked to the response tokens of each row, and reduced as a per-token mean across the mini-batch.

## File layout

| File | Role |
| --- | --- |
| [`main_opsd.py`](./main_opsd.py) | Hydra entry point; spawns the Ray TaskRunner with `RayOPSDTrainer`. |
| [`opsd_ray_trainer.py`](./opsd_ray_trainer.py) | Subclass of `RayPPOTrainer` overriding `fit` with the OPSD loop and `_build_teacher_tensors` for the teacher inputs. |
| [`opsd_fsdp_workers.py`](./opsd_fsdp_workers.py) | Subclass of `AsyncActorRolloutRefWorker` that wires the colocated teacher into the actor and exposes `update_actor_opsd`. |
| [`opsd_dp_actor.py`](./opsd_dp_actor.py) | Subclass of `DataParallelPPOActor` adding the OPSD update step (`update_policy_opsd`) and a full-vocab `_forward_logits` slice. |
| [`opsd_core_algos.py`](./opsd_core_algos.py) | Stand-alone `generalized_jsd_loss`. |
| [`opsd_utils.py`](./opsd_utils.py) | `validate_opsd_config`: pre-flight checks. |
| [`config/opsd_trainer.yaml`](./config/opsd_trainer.yaml) | Defaults extending `ppo_trainer.yaml`. |
