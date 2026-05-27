# Flappy-Bird-RL

A minimal Deep Q-Network (DQN) implementation that learns to play Flappy Bird using Gymnasium + flappy_bird_gymnasium and PyTorch. The project includes a training loop with experience replay, a target network, and simple checkpointing/logging.

## Features
- Simple fully-connected `DQN` model for state-based inputs
- FIFO experience replay buffer
- Epsilon-greedy exploration with configurable decay
- Target-network synchronization for stable updates
- Config-driven hyperparameters via `parameters.yaml`
- Save/load best model checkpoints to `runs/`

## Quick start

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Activate the venv (platform dependent)
# Windows:
.venv\\Scripts\\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

2. Train:
```bash
python agent.py flappybirdv0 --train
```

3. Run a trained model (visual mode):
```bash
python agent.py flappybirdv0
```

## How it works (concise)
- `agent.py` loads hyperparameters from `parameters.yaml`, builds a `DQN` policy and (when training) a target network.
- Episodes are collected with epsilon-greedy action selection; experiences are stored in `ReplayMemory`.
- Mini-batches are sampled to compute TD-targets: reward + gamma * max_a' Q_next (zeroed for terminal states).
- Loss is MSE between current Q and target Q; optimizer is Adam with `alpha` as learning rate.
- Target network is synced every `network_sync_rate` steps; best episode policies are checkpointed in `runs/`.

## Configuration
See `parameters.yaml` to change hyperparameters such as:
- `epsilon_init`, `epsilon_min`, `epsilon_decay`
- `replay_memory_size`, `mini_batch_size`
- `network_sync_rate`, `alpha` (learning rate), `gamma` (discount)
- `reward_threshold` — episode cap used in the training loop

## Files of interest
- [agent.py](agent.py) — training/evaluation loop, device selection, save/load logic
- [dqn.py](dqn.py) — `DQN` network (simple MLP)
- [experience_replay.py](experience_replay.py) — `ReplayMemory` FIFO buffer
- [game_flappy_bird.py](game_flappy_bird.py) — manual keyboard-play demo using pygame
- [parameters.yaml](parameters.yaml) — hyperparameter presets
- `runs/` — saved weights and logs
