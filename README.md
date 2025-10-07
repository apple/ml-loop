# Reinforcement Learning for Long-Horizon Interactive LLM Agents

This repository contains a reference implementation of **Leave-One-Out PPO (LOOP)**, as well as training and evaluation code for the [AppWorld](https://appworld.dev/) benchmark.

This software project accompanies the research paper, <br/>**[Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/abs/2502.01600)**,
*Kevin Chen, Marco Cusumano-Towner, Brody Huval, Aleksei Petrenko, Jackson Hamburger, Vladlen Koltun, and Philipp Krähenbühl*

Interactive digital agents (IDAs) leverage APIs of stateful digital environments to perform tasks in response to user requests. While IDAs powered by instruction-tuned large language models (LLMs) can react to feedback from interface invocations in multi-step exchanges, they have not been trained in their respective digital environments. Prior methods accomplish less than half of tasks in sophisticated benchmarks such as AppWorld. We present a reinforcement learning (RL) approach that trains IDAs directly in their target environments. We formalize this training as a partially observable Markov decision process and derive LOOP, a data- and memory-efficient variant of proximal policy optimization. LOOP uses no value network and maintains exactly one copy of the underlying LLM in memory, making its implementation straightforward and as memory-efficient as fine-tuning a single LLM. A 32-billion-parameter agent trained with LOOP in the AppWorld environment outperforms the much larger OpenAI o1 agent by 9 percentage points (15% relative). To our knowledge, this is the first reported application of RL to IDAs that interact with a stateful, multi-domain, multi-app environment via direct API calls. Our analysis sheds light on the effectiveness of RL in this area, showing that the agent learns to consult the API documentation, avoid unwarranted assumptions, minimize confabulation, and recover from setbacks.

## Installation

We use poetry>=2.1 to manage dependencies.
Install poetry in your preferred Python>=3.12 environment and then proceed to install dependencies:

```bash
poetry install
```

Install AppWorld into a separate virtual env to avoid dependency conflicts between AppWorld and vLLM:

```bash
python -m virtualenv appworld-env && appworld-env/bin/pip install click==8.2.1 appworld && appworld-env/bin/appworld install
```

Set the envvar `APPWORLD_ROOT` to the full desired path for the AppWorld data directory and download the data:

```bash
export APPWORLD_ROOT=<...>
appworld-env/bin/appworld download data --root $APPWORLD_ROOT
```

## Model Training

Execute the following command optimized for a single 8-GPU node (8xA100, 8xH100, or similar), replacing the placeholder values appropriately.
For Qwen 2.5 32B post-training we assume 8x80 GB GPUs. Smaller models can be trained with less memory.
In this example, GPUs [0..3] are used for training and GPUs [4..7] are used for inference.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_TOKEN=<...> \
WANDB_API_KEY=<...> \
APPWORLD_ROOT=<...> \
PATH=<path_to_repo>/appworld-env/bin:$PATH \
accelerate launch \
  --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
  --num_processes=4 \
  ./phi_agents/rl/train.py \
  +global@_global_=appworld \
  rl/gpu_allocation=four_learn_four_infer \
  llm=qwen_2_5_32b_train \
  experiment_name=2025_appworld_Q32B \
  wandb.enable=True \
  wandb.group=2025_appworld_Q32B_v1 \
  wandb.project=appworld_Q32B \
  rl.params.total_iterations=200 \
  rl.max_ckpts=200 \
  rl.scenario_runner.appworld_config.env.max_interactions=40 \
  rl.eval.overrides.rl.scenario_runner.appworld_config.env.max_interactions=40 \
  rl.num_scenario_runners=32 \
  rl.params.scenarios_per_iteration=40 \
  rl.params.minibatch_size=32 \
  rl.params.rollouts_per_scenario=6 \
  rl.scenario_sampler.dataset_name=train_difficulty_1_and_train_difficulty_2 \
  rl.learning_max_seq_len=32000 \
  rl.rollouts_fraction=0.9 \
  rl.rollouts_per_scenario_fraction=0.75
```

Before the 1st iteration of training, the script performs a memory stress test to ensure that
the longest trajectory will fit into memory during training.
In case of memory pressure, decrease rl.learning_max_seq_len to 24000 (very modest impact on final performance).
If this is not sufficient, take a look at the comments in accelerate_config.yaml, specifically `fsdp_offload_params: true` parameter allows you to trade throughput for memory.

## Evaluation

The training script will continuously evaluate the latest policy on the validation (`dev`) split.
In addition to that, you can use the following standalone script to evaluate any checkpoint on a given task split:

```bash
CUDA_VISIBLE_DEVICES=<...> \
APPWORLD_ROOT=<...> \
PATH=<path_to_repo>/appworld-env/bin:$PATH \
python -m scripts.run_appworld_inference \
experiment_name=appworld_eval_test_normal \
llm=qwen_2_5_32b_eval \
llm.adapter_path=<your-checkpoint> \
scenario_sampler.dataset_name=test_normal
```

Change `dataset_name` to evaluate on a different AppWorld split.
By default, this will save the episode rollouts to `$APPWORLD_ROOT/experiments/outputs/<experiment_name>`.

The following script analyzes the rollouts, calculates and logs success rate metrics (SGC, TGC):

```bash
APPWORLD_ROOT=<...> \
PATH=<path_to_repo>/appworld-env/bin:$PATH \
python -m scripts.appworld_eval_parse_and_log \
experiment_name=appworld_eval_test_normal \
scenario_sampler.dataset_name=test_normal
```

## Scaling

The system was originally designed to run on a distributed cluster, e.g. training, inference, and eval
processes can each use multiple nodes for maximum throughput.

In this repository, we provide a single-node version of the training system that sets up a local Ray cluster
with three distinct Ray workers: trainer (train.py, includes the main training loop), vllm server (vllm_server.py, inference), and the evaluator (eval.py).
With minimal refactoring, these components can be executed on a distributed cluster (see `connect_ray_cluster()`).

Note that the local single-node 8-GPU setup is sufficient for reproducing the training results.

## Debugging

A minimal configuration can be used for local debugging on a single GPU with a smaller model:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_TOKEN=<...> \
WANDB_API_KEY=<...> \
APPWORLD_ROOT=<...> \
PATH=<path_to_repo>/appworld-env/bin:$PATH \
accelerate launch \
  --config_file ./phi_agents/rl/conf/accelerate_config.yaml \
  --num_processes=1 \
  ./phi_agents/rl/train.py \
  +global@_global_=appworld \
  rl/gpu_allocation=single_gpu \
  llm=qwen_2_5_7b_train \
  experiment_name=2025_appworld_debug \
  wandb.enable=False \
  rl.eval.enable=False \
  rl.params.total_iterations=2 \
  rl.scenario_runner.appworld_config.env.max_interactions=5 \
  rl.num_scenario_runners=1 \
  rl.params.scenarios_per_iteration=2 \
  rl.params.minibatch_size=2 \
  rl.params.rollouts_per_scenario=3 \
  rl.scenario_sampler.dataset_name=train \
  rl.learning_max_seq_len=20000 \
  llm.vllm_class.max_new_tokens=2000
```

On a GPU with 40GB VRAM or less, inference and training typically do not fit in memory at the same time.
Thus in this configuration we tear down the vLLM server and restart it again between iterations to save memory.
Note that this is slow and is only meant for debugging, not for actual training.

Alternatively:

* You can manipulate `rl.inference_requires_memory_gb` and `rl.learning_requires_memory_gb` (e.g. set both to under 50% of your GPU's VRAM)
This will prompt the system to run vLLM and training at the same time on the same GPU, splitting memory between the two.
Such configuration is useful for debugging on a 80GB GPU (A100, H100) with smaller models.

* The system can connect to an existing vLLM server, this significantly speeds up startup and debugging (controlled by `llm.vllm_server.allow_connect_to_existing`)

* Specify `rl/gpu_allocation=one_inference_one_learning` for efficient debugging on a 2-GPU machine.

## Citation

```bibtex
@inproceedings{chen2025loop,
  author       = {Kevin Chen and
                  Marco Cusumano-Towner and
                  Brody Huval and
                  Aleksei Petrenko and
                  Jackson Hamburger and
                  Vladlen Koltun and
                  Philipp Kr\"ahenb\"uhl},
  title        = {Reinforcement Learning for Long-Horizon Interactive LLM Agents},
  booktitle    = {arXiv},
  year         = {2025},
}
```

## License

This sample code is released under the [LICENSE](LICENSE) terms.

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [Acknowledgements](ACKNOWLEDGEMENTS.md) for more details.

Please check the paper for a complete list of references and datasets used in this work.
