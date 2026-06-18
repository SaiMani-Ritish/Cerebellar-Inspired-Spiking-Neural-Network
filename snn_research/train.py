"""
Training loops and the 100-episode experiment runner.

  - train_mnist_ann / train_mnist_snn : MNIST sanity-check trainers
  - train_snn_controller              : CartPole A2C + STDP with endpoint shaping
  - run_experiment                    : evaluation harness
  - main()                            : CLI entry point

Algorithm: Advantage Actor-Critic (A2C)
  - Actor:   SNNController (LIF + surrogate gradient)
  - Critic:  ValueNetwork  (plain MLP, raw obs → V(s))
  - Advantage: A_t = G_t - V(s_t)   (variance-reduced return)
  - Entropy bonus to discourage premature policy collapse
  - Reward shaping: terminal penalty -endpoint_lambda * |final_pos|
"""

import os
import argparse
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import gymnasium as gym

try:
    from .model import (
        SimpleANN,
        SimpleSNN,
        QuinticPlanner,
        SNNController,
        ValueNetwork,
    )
    from .data import state_to_spikes, encode_batch
    from .evaluate import log_episode
except ImportError:  # pragma: no cover
    from model import (
        SimpleANN,
        SimpleSNN,
        QuinticPlanner,
        SNNController,
        ValueNetwork,
    )
    from data import state_to_spikes, encode_batch
    from evaluate import log_episode


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# A2C hyper-parameters (can be overridden via train_snn_controller kwargs)
ENTROPY_COEFF = 0.01   # H(pi) bonus — prevents collapse to single action
VALUE_COEFF   = 0.5    # critic loss weight
ENDPOINT_LAMBDA = 5.0  # penalty coefficient for |final cart position|
MAX_GRAD_NORM = 0.5    # gradient clipping for stability


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# MNIST training (unchanged — kept as sanity-check baselines)
# ---------------------------------------------------------------------------

def _mnist_loaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, transform=transform)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )


def train_mnist_ann(
    model: SimpleANN,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> dict:
    """Train SimpleANN on MNIST. Returns {epoch: {loss, accuracy}}."""
    device = device or _get_device()
    model = model.to(device)
    train_loader, test_loader = _mnist_loaders()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: dict = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out, _ = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        acc = _eval_mnist(model, test_loader, device)
        history[epoch] = {"loss": total_loss / len(train_loader), "accuracy": acc}
        print(f"[ANN] Epoch {epoch+1}/{epochs}  loss={history[epoch]['loss']:.4f}  acc={acc:.2%}")

    return history


def train_mnist_snn(
    model: SimpleSNN,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> dict:
    """Train SimpleSNN on MNIST with surrogate gradient. Returns history dict."""
    device = device or _get_device()
    model = model.to(device)
    train_loader, test_loader = _mnist_loaders()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: dict = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out, _ = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        acc = _eval_mnist(model, test_loader, device)
        history[epoch] = {"loss": total_loss / len(train_loader), "accuracy": acc}
        print(f"[SNN] Epoch {epoch+1}/{epochs}  loss={history[epoch]['loss']:.4f}  acc={acc:.2%}")

    return history


@torch.no_grad()
def _eval_mnist(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out, _ = model(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# A2C CartPole training
# ---------------------------------------------------------------------------

def _compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Discounted return G_t = r_t + γ*r_{t+1} + … for each timestep."""
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_snn_controller(
    model: SNNController,
    n_episodes: int = 500,
    gamma: float = 0.99,
    lr_actor: float = 5e-4,
    lr_critic: float = 1e-3,
    endpoint_lambda: float = ENDPOINT_LAMBDA,
    entropy_coeff: float = ENTROPY_COEFF,
    device: torch.device | None = None,
) -> list[float]:
    """
    Train SNNController on CartPole-v1 using A2C + endpoint reward shaping.

    Actor (SNNController):  surrogate-gradient policy gradient on advantages.
    Critic (ValueNetwork):  MSE loss against Monte-Carlo returns.
    STDP timing layer:      Hebbian update after each episode.
    Reward shaping:         terminal step penalised by endpoint_lambda * |pos|.

    Returns list of episode rewards.
    """
    device = device or _get_device()
    model = model.to(device)

    critic = ValueNetwork(obs_dim=4).to(device)

    opt_actor  = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr_actor
    )
    opt_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    env = gym.make("CartPole-v1")
    reward_history: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)

        log_probs:     list[torch.Tensor] = []
        entropies:     list[torch.Tensor] = []
        values:        list[torch.Tensor] = []
        rewards:       list[float]        = []
        pre_spikes_ep: list[torch.Tensor] = []
        post_spikes_ep:list[torch.Tensor] = []

        model.train()
        done = False

        while not done:
            # --- Critic: value estimate for current obs ---
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            value = critic(obs_t)   # (1,)

            # --- Actor: spike-encoded forward pass ---
            spikes_list = state_to_spikes(obs)
            spike_t = torch.tensor([spikes_list], dtype=torch.float32, device=device)

            mem1     = model.lif1.init_leaky()
            mem2     = model.lif2.init_leaky()
            mem_stdp = model.lif_stdp.init_leaky()

            step_pre:    list[torch.Tensor] = []
            step_post:   list[torch.Tensor] = []
            out_spk_rec: list[torch.Tensor] = []

            for _ in range(model.num_steps):
                cur1 = model.fc1(spike_t)
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur2, mem2)
                cur_stdp = spk2 @ model.stdp_weights
                spk_stdp, mem_stdp = model.lif_stdp(cur_stdp, mem_stdp)

                step_pre.append(spk2.detach())
                step_post.append(spk_stdp.detach())
                out_spk_rec.append(spk_stdp)

            rate   = torch.stack(out_spk_rec, dim=0).mean(dim=0)
            logits = model.fc_out(rate)
            probs  = torch.softmax(logits, dim=1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value.squeeze(0))
            pre_spikes_ep.extend(step_pre)
            post_spikes_ep.extend(step_post)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        # --- Endpoint reward shaping: penalise final cart position ---
        final_pos = float(obs[0])
        rewards[-1] -= endpoint_lambda * abs(final_pos)

        # --- Monte-Carlo returns and advantage estimation ---
        returns_raw = _compute_returns(rewards, gamma)
        returns_t   = torch.tensor(returns_raw, dtype=torch.float32, device=device)
        values_t    = torch.stack(values)          # (T,) — differentiable

        advantages  = returns_t - values_t.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Actor (policy) loss: -E[log_pi(a|s) * A_t] - entropy bonus ---
        actor_loss = torch.tensor(0.0, device=device)
        for lp, adv, ent in zip(log_probs, advantages, entropies):
            actor_loss = actor_loss - lp * adv - entropy_coeff * ent

        # --- Critic (value) loss: MSE of V(s) vs Monte-Carlo G_t ---
        critic_loss = VALUE_COEFF * nn.functional.mse_loss(values_t, returns_t)

        # --- Joint update ---
        total_loss = actor_loss + critic_loss
        opt_actor.zero_grad()
        opt_critic.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        opt_actor.step()
        opt_critic.step()

        # --- STDP update on timing layer (Hebbian, separate from grad) ---
        model.apply_stdp(pre_spikes_ep, post_spikes_ep)

        ep_reward = sum(rewards)
        reward_history.append(ep_reward)
        if ep % 20 == 0:
            print(f"[SNN-Ctrl] Episode {ep}/{n_episodes}  reward={ep_reward:.0f}")

    env.close()
    return reward_history


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def run_experiment(
    controller: Callable,
    controller_name: str,
    n_episodes: int = 100,
) -> pd.DataFrame:
    """
    Evaluate a controller on CartPole-v1 for n_episodes.

    controller must be callable: (action, _) = controller(obs)
    Saves results CSV to outputs/.
    """
    env = gym.make("CartPole-v1")
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)

        if hasattr(controller, "reset"):
            controller.reset(obs)

        positions: list[float] = []
        done = False

        while not done:
            action, _ = controller(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            positions.append(float(obs[0]))
            done = terminated or truncated

        metrics = log_episode(controller_name, positions)
        results.append(metrics)
        if ep % 10 == 0:
            print(f"  [{controller_name}] Episode {ep}/{n_episodes} done")

    env.close()
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUTS_DIR, f"{controller_name}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SNN Research Experiment Runner")
    parser.add_argument("--episodes",       type=int,  default=100, help="Eval episodes per controller")
    parser.add_argument("--train-episodes", type=int,  default=500, help="A2C training episodes for SNN controller")
    parser.add_argument("--skip-training",  action="store_true",    help="Skip SNN training, use random weights")
    args = parser.parse_args()

    device = _get_device()
    print(f"Device: {device}")

    # --- Quintic Polynomial Planner (analytical baseline) ---
    print("\n=== Quintic Polynomial Planner ===")
    planner = QuinticPlanner()
    df_poly = run_experiment(planner, "quintic_polynomial", n_episodes=args.episodes)

    # --- SNN controller (A2C, surrogate-grad only, no STDP) ---
    print("\n=== SNN (surrogate-grad + A2C, no STDP) ===")
    snn_no_stdp = SNNController(beta=0.9, num_steps=25)
    snn_no_stdp.to(device)

    if not args.skip_training:
        print(f"Training SNN (no STDP) for {args.train_episodes} episodes…")
        snn_no_stdp.stdp_params = {
            "A_plus": 0.0, "A_minus": 0.0,
            "tau_plus": 20.0, "tau_minus": 20.0,
        }
        train_snn_controller(snn_no_stdp, n_episodes=args.train_episodes, device=device)

    snn_no_stdp.eval()
    df_snn = run_experiment(snn_no_stdp.act, "snn_surrogate_only", n_episodes=args.episodes)

    # --- SNN + STDP timing (A2C + Hebbian) ---
    print("\n=== SNN + STDP timing (A2C) ===")
    snn_stdp = SNNController(beta=0.9, num_steps=25)
    snn_stdp.to(device)

    if not args.skip_training:
        print(f"Training SNN + STDP for {args.train_episodes} episodes…")
        train_snn_controller(snn_stdp, n_episodes=args.train_episodes, device=device)

    snn_stdp.eval()
    df_stdp = run_experiment(snn_stdp.act, "snn_stdp", n_episodes=args.episodes)

    # --- Summary ---
    print("\n=== Summary (mean ± std) ===")
    for name, df in [("Quintic", df_poly), ("SNN-only", df_snn), ("SNN+STDP", df_stdp)]:
        print(f"\n{name}:")
        for col in ["peak_jerk", "endpoint_err", "osc_amp"]:
            if col in df.columns:
                print(f"  {col:15s} = {df[col].mean():.4f} ± {df[col].std():.4f}")


if __name__ == "__main__":
    main()
