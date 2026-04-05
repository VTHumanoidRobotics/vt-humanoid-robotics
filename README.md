# VT Humanoid Robotics — Controls & Software: Reinforcement Learning & Policy


> Forked from [Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite) by UC Berkeley's Hybrid Robotics Lab.

This is the working repository for the **Controls & Software subteam** of VT Humanoid Robotics. It contains the Berkeley Humanoid Lite codebase (IsaacLab training environments, robot descriptions, low-level deployment code) along with our team's onboarding docs and guides.

---

## What's in this repo

| Folder | What it contains |
|---|---|
| `source/berkeley_humanoid_lite/` | IsaacLab environment and task definitions for RL training |
| `source/berkeley_humanoid_lite_assets/` | Robot description files (URDF, MJCF, USD) |
| `source/berkeley_humanoid_lite_lowlevel/` | Low-level code that runs on the physical robot |
| `scripts/` | Entry points for training, visualization, and deployment |
| `configs/` | Policy configuration files for sim2sim and sim2real |
| `checkpoints/` | Saved policy checkpoints |
| `docs/` | Our team's onboarding and setup guides |

---

## Quick Start

### 1. Install Linux

You need Ubuntu 22.04 or 24.04. If you're on Windows, you have two paths:

- **WSL2** — fine for Python, Gymnasium, Stable Baselines3, and general RL work. Install with `wsl --install -d Ubuntu-24.04` in PowerShell (admin).
- **Dual-boot** — needed if you ever want to run Isaac Sim locally. Not required right now (see GPU note below).

After getting into Ubuntu (WSL or native), update everything:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install cmake build-essential git
```

### 2. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Accept the license, say yes to conda init
# Restart your terminal after install
conda --version   # should print something like conda 24.x.x
```

### 3. Create the conda environment

```bash
conda create -yn berkeley-humanoid-lite python=3.11
conda activate berkeley-humanoid-lite
```

> **Always activate this environment** before doing any work in this repo. If your terminal prompt doesn't show `(berkeley-humanoid-lite)` at the start, run `conda activate berkeley-humanoid-lite` first.

### 4. Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify it installed:

```bash
python -c "import torch; print(torch.__version__)"
```

> **Note on CUDA:** If `torch.cuda.is_available()` returns `False`, that's expected if you're on WSL or don't have an NVIDIA GPU. PyTorch will still work on CPU — training will just be slower. See the GPU note below.

### 5. Clone this repo and install

```bash
git clone https://github.com/VTHumanoidRobotics/vt-humanoid-robotics.git
cd vt-humanoid-robotics

# Initialize submodules (robot descriptions, low-level code)
git submodule update --init

# Install the Berkeley Humanoid Lite extensions
pip install -e ./source/berkeley_humanoid_lite/
pip install -e ./source/berkeley_humanoid_lite_assets/
pip install -e ./source/berkeley_humanoid_lite_lowlevel/

# Install additional dependencies
pip install -r requirements.txt
```

### 6. Install RL libraries

These are the core RL tools you'll use for learning and experimentation:

```bash
pip install gymnasium stable-baselines3[extra] numpy matplotlib scipy
pip install tensorboard wandb mujoco tqdm pandas
```

Optional (for specific demos):

```bash
pip install "gymnasium[classic-control]"   # CartPole, MountainCar
pip install "gymnasium[mujoco]"            # MuJoCo continuous control
pip install "gymnasium[box2d]"             # LunarLander, BipedalWalker
```

### 7. Verify your setup

```bash
# Test that the Berkeley Humanoid Lite package is importable
python -c "import berkeley_humanoid_lite; print('Berkeley Humanoid Lite OK')"

# Test that Gymnasium + SB3 work
python -c "from stable_baselines3 import PPO; import gymnasium as gym; env = gym.make('CartPole-v1'); model = PPO('MlpPolicy', env, verbose=1); model.learn(total_timesteps=10000); print('All good!')"
```

---

## A note about GPUs and Isaac Lab

The Berkeley Humanoid Lite training pipeline uses **NVIDIA Isaac Lab**, which requires an NVIDIA RTX GPU with RT cores (RTX 3070+, ideally 4070+) and a native Ubuntu install (not WSL). **We don't currently have access to this hardware as a team, so don't worry about installing Isaac Sim or Isaac Lab right now.**

What this means in practice:

- You **can** explore the codebase, read the task definitions, and understand how the environments are structured — all without a GPU.
- You **can** learn RL fundamentals using Gymnasium + Stable Baselines3 on CPU (or your own GPU if you have one).
- You **can** run sim2sim validation in MuJoCo, which doesn't require Isaac Sim.
- When we're ready to train locomotion policies, we'll use shared compute resources (lab machines, cloud GPU instances, etc.).

For now, focus on understanding the RL pipeline and getting comfortable with Gymnasium, Stable Baselines3, and Python.

---

## Training (when you have GPU access)

Two tasks are defined in the codebase:

| Task | DOF | Description |
|---|---|---|
| `Velocity-Berkeley-Humanoid-Lite-v0` | 22 | Full body control (legs + arms) |
| `Velocity-Berkeley-Humanoid-Lite-Biped-v0` | 12 | Legs only |

```bash
# Train (headless, no GUI)
python ./scripts/rsl_rl/train.py --task Velocity-Berkeley-Humanoid-Lite-v0 --headless

# Visualize a trained policy
python ./scripts/rsl_rl/play.py --task Velocity-Berkeley-Humanoid-Lite-v0 --num_envs 16

# Monitor training
tensorboard --logdir logs/
```

Training defaults to 6000 iterations (~2 hours on a modern GPU). The play script exports the policy to ONNX for deployment.

---

## RL Intro Resources

If you're new to reinforcement learning, check out these resources to get started:

- [**Kelly's HSSP RL Research Demos**](https://github.com/kelly-castillo/hssp-rl-research) — 5 interactive RL demos built with CleanRL during the Hokie Summer Scholars Program. Great starting point for understanding PPO, curriculum learning, multi-agent RL, safe RL, and exploration strategies.
- [Gymnasium Documentation](https://gymnasium.farama.org/) — the standard RL environment API
- [Stable Baselines3 Docs](https://sb3-contrib.readthedocs.io/en/master/) — reliable RL algorithm implementations
- [Spinning Up in Deep RL](https://spinningup.openai.com/) — OpenAI's intro to deep RL (excellent theory + code)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file RL implementations, great for reading and understanding algorithms

---

## Useful links

- [Berkeley Humanoid Lite Documentation](https://berkeley-humanoid-lite.gitbook.io/docs)
- [Berkeley Humanoid Lite Paper (arXiv)](https://arxiv.org/abs/2504.17249)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/)
- [Isaac Lab Pip Install Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- [Our Team's Onboarding Guide](docs/onboarding.md)

---

## Contributing

This repo is maintained by the Controls & Software subteam. If you're a team member:

1. Create a branch for your work: `git checkout -b your-name/feature-description`
2. Make your changes and commit with clear messages
3. Push and open a pull request
4. Get at least one review before merging

Questions? Reach out in the Controls/Software Channel or Kelly Castillo on Slack.
