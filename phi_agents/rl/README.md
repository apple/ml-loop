# RL Trainer

## Diagrams for different GPU allocations

### learning_gpus: [0,1,2,3,4,5] and inference_gpus: [6,7]

There should be 6 processes, one for each learning GPU.
Here proc 0 will also be responsible for managing two vLLM servers (which run in their own processes).

This type of configuration where GPUs are not shared between learning and inference
was designed for `rl.async_rollouts=True`, meaning new rollouts are collected during the learning
phase using the previous version of the policy. It also allows each type of process to use all available GPU memory.

NOTE: vLLM server does not need to be restarted every training iteration.

### learning_gpus: [0,1,2,3,4,5,6,7] and inference_gpus: [0,1,2,3,4,5,6,7]

There should be 8 learner processes, one for each learning GPU.
Proc 0 will also be responsible for managing eight vLLM servers (vLLM servers run in their own processes).

This configuration can be used with `rl.async_rollouts=False`, guaranteeing synchronous collection
of new rollouts before each learning phase. With `rl.async_rollouts=False`, rollouts have zero policy-lag
before the first SGD step.

If `learning_gpus` and `inference_gpus` overlap, like here, and vLLM and learner processes don't fit in memory together,
then the vLLM server will need to be restarted every iteration (see the diagram below).
The expected memory usage is specified by: `rl.inference_requires_memory_gb` and `rl.learning_requires_memory_gb`.

```
process |   Rollout collection                        |  Gradients        |
proc 0  |   start vLLM;                   stop vLLM;  |   using GPU 0     |
proc 1  |                                             |   using GPU 1     |
...     |                                             |   ...             |
proc 7  |                                             |   using GPU 7     |
vLLM    |               running (8 GPUs);             |   none            |
```

When GPU memory is enough to fit both vLLM and the learner process, vLLM servers won't be restarted.

## Training different AppWorld agents

The RL code does not restrict you to use a single agent.
To implement a new AppWorld agent for use with RL, create a new class for your agent.
The class must implement the interface `AppworldAgent` defined in `phi_agents/evals/appworld_evals.py`.
Then, create a config file for your agent in `phi_agents/rl/conf/appworld/agent/`.
See `default.yaml` for an example.
Note that the class of your agent is specified in the `_target_` field.
Suppose that your agent's config file is in `my_agent.yaml`.
Then, when running or submitting a training job, add the following to the command line:

```bash
appworld/agent@rl.scenario_runner.appworld_config.agent=my_agent
```
