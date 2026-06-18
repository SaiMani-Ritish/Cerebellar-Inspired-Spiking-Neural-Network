"""
All controller / model classes:
  - SimpleANN:      dense feedforward baseline for MNIST comparison
  - SimpleSNN:      2-layer LIF network for MNIST sanity check
  - QuinticPlanner: analytical polynomial trajectory planner (baseline)
  - SNNController:  LIF hidden layers + STDP timing layer (research contribution)
  - stdp_update:    standalone Hebbian weight update
  - ValueNetwork:   A2C critic MLP operating on raw CartPole observations
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

try:
    # When executed as a package module (e.g. `python -m snn_research.model`)
    from .data import state_to_spikes, encode_batch
except ImportError:  # pragma: no cover
    # When executed as a script from this directory (e.g. `python model.py`)
    from data import state_to_spikes, encode_batch


# ---------------------------------------------------------------------------
# MNIST models
# ---------------------------------------------------------------------------

class SimpleANN(nn.Module):
    """Dense ReLU baseline for MNIST. Used for FLOP comparison against SimpleSNN."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x), 0  # 0 spikes (ANN)


class SimpleSNN(nn.Module):
    """2-layer LIF network for MNIST. Returns (logits, total_spike_count)."""

    def __init__(self, beta: float = 0.9, num_steps: int = 25):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(784, 512)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spike_count = 0

        spk2_rec = []
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spike_count = spike_count + spk1.sum() + spk2.sum()
            spk2_rec.append(spk2)

        spk2_stack = torch.stack(spk2_rec, dim=0)
        out = self.fc3(spk2_stack.sum(dim=0))
        return out, spike_count


# ---------------------------------------------------------------------------
# Quintic polynomial planner (analytical baseline)
# ---------------------------------------------------------------------------

def quintic_coeffs(
    p0: float, pf: float, v0: float, T: float
) -> list[float]:
    """
    Solve for quintic polynomial coefficients from boundary conditions:
      pos(0)=p0, vel(0)=v0, acc(0)=0
      pos(T)=pf, vel(T)=0,  acc(T)=0
    """
    a0, a1, a2 = p0, v0, 0.0
    M = np.array([
        [T**3,     T**4,      T**5],
        [3*T**2,   4*T**3,    5*T**4],
        [6*T,      12*T**2,   20*T**3],
    ])
    b = np.array([pf - a0 - a1 * T, -a1, 0.0])
    a3, a4, a5 = np.linalg.solve(M, b)
    return [a0, a1, a2, float(a3), float(a4), float(a5)]


def compute_jerk(coeffs: list[float], t: float) -> float:
    """Jerk from quintic: J(t) = 6*a3 + 24*a4*t + 60*a5*t^2."""
    _, _, _, a3, a4, a5 = coeffs
    return 6 * a3 + 24 * a4 * t + 60 * a5 * t**2


def quintic_position(coeffs: list[float], t: float) -> float:
    a0, a1, a2, a3, a4, a5 = coeffs
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5


def quintic_velocity(coeffs: list[float], t: float) -> float:
    a0, a1, a2, a3, a4, a5 = coeffs
    return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4


class QuinticPlanner:
    """
    CartPole controller that follows a quintic polynomial trajectory to
    bring the cart to position 0 with zero velocity.

    Callable interface: action, spike_count = planner(obs)
    """

    def __init__(self, T: float = 2.0, dt: float = 0.02, Kp: float = 5.0, Kd: float = 2.0):
        self.T = T
        self.dt = dt
        self.Kp = Kp
        self.Kd = Kd
        self._coeffs: Optional[list[float]] = None
        self._t = 0.0
        self._flop_per_step = 20  # analytical ops per step (cheap)

    def reset(self, obs: np.ndarray):
        p0, v0 = float(obs[0]), float(obs[1])
        self._coeffs = quintic_coeffs(p0, 0.0, v0, self.T)
        self._t = 0.0

    def __call__(self, obs: np.ndarray) -> tuple[int, int]:
        if self._coeffs is None:
            self.reset(obs)

        self._t += self.dt
        t = min(self._t, self.T)

        desired_pos = quintic_position(self._coeffs, t)
        desired_vel = quintic_velocity(self._coeffs, t)

        pos, vel = float(obs[0]), float(obs[1])
        force = self.Kp * (desired_pos - pos) + self.Kd * (desired_vel - vel)

        action = 1 if force > 0 else 0
        return action, self._flop_per_step


# ---------------------------------------------------------------------------
# STDP update rule
# ---------------------------------------------------------------------------

def stdp_update(
    weights: torch.Tensor,
    pre_spikes: list[torch.Tensor],
    post_spikes: list[torch.Tensor],
    A_plus: float = 0.01,
    A_minus: float = 0.01,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    Spike-Timing Dependent Plasticity weight update.

    pre fires BEFORE post  -> strengthen (LTP, A_plus)
    pre fires AFTER post   -> weaken     (LTD, A_minus)

    Operates on recorded spike trains (lists of tensors, one per timestep).
    """
    dW = torch.zeros_like(weights)
    device = weights.device

    for t_pre, spk_pre in enumerate(pre_spikes):
        for t_post, spk_post in enumerate(post_spikes):
            pre_active = spk_pre.sum() > 0
            post_active = spk_post.sum() > 0
            if pre_active and post_active:
                delta_t = (t_post - t_pre) * dt
                if delta_t > 0:
                    dW += A_plus * torch.exp(torch.tensor(-delta_t / tau_plus, device=device))
                elif delta_t < 0:
                    dW -= A_minus * torch.exp(torch.tensor(delta_t / tau_minus, device=device))
    return weights + dW


# ---------------------------------------------------------------------------
# SNN Controller (research contribution)
# ---------------------------------------------------------------------------

class SNNController(nn.Module):
    """
    Cerebellar-inspired SNN controller for CartPole:
      Input(8 spike dims) -> LIF hidden(64) -> LIF hidden(32) -> STDP timing(16) -> output(2)

    Hidden layers trained with surrogate gradient backprop.
    STDP timing layer updated with Hebbian STDP (no gradient).
    Output decoded via spike rate over the time window.
    """

    def __init__(
        self,
        beta: float = 0.9,
        num_steps: int = 25,
        stdp_params: Optional[dict] = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Hidden LIF layers (surrogate gradient trained)
        self.fc1 = nn.Linear(8, 64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # STDP timing layer (Hebbian, not gradient)
        self.stdp_weights = nn.Parameter(
            torch.randn(32, 16) * 0.1, requires_grad=False
        )
        self.lif_stdp = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Output projection
        self.fc_out = nn.Linear(16, 2)

        self.stdp_params = stdp_params or {
            "A_plus": 0.01,
            "A_minus": 0.01,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
        }

        self._device = None
        self._flops_per_step: Optional[int] = None

    @property
    def device(self):
        return next(self.parameters()).device

    def _estimate_flops(self) -> int:
        """Rough spike-gated MAC count per forward pass."""
        return (8 * 64 + 64 * 32 + 32 * 16 + 16 * 2) * self.num_steps

    def forward(self, spike_input: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Args:
            spike_input: (batch, 8) encoded spike vector.
        Returns:
            (action_logits, total_spike_count)
        """
        batch = spike_input.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_stdp = self.lif_stdp.init_leaky()

        spike_count = 0
        pre_rec: list[torch.Tensor] = []
        post_rec: list[torch.Tensor] = []
        out_spk_rec: list[torch.Tensor] = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(spike_input)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # STDP timing layer (manual matmul, no autograd on weights)
            cur_stdp = spk2 @ self.stdp_weights
            spk_stdp, mem_stdp = self.lif_stdp(cur_stdp, mem_stdp)

            pre_rec.append(spk2.detach())
            post_rec.append(spk_stdp.detach())
            out_spk_rec.append(spk_stdp)

            spike_count = spike_count + spk1.sum() + spk2.sum() + spk_stdp.sum()

        # Rate decoding: average spikes over time
        rate = torch.stack(out_spk_rec, dim=0).mean(dim=0)
        logits = self.fc_out(rate)

        return logits, int(spike_count.item())

    def apply_stdp(
        self,
        pre_spikes: list[torch.Tensor],
        post_spikes: list[torch.Tensor],
    ):
        """Apply STDP update to the timing layer weights (in-place, no grad)."""
        with torch.no_grad():
            self.stdp_weights.copy_(
                stdp_update(
                    self.stdp_weights,
                    pre_spikes,
                    post_spikes,
                    **self.stdp_params,
                )
            )

    def act(self, obs: np.ndarray) -> tuple[int, int]:
        """
        Callable controller interface for run_experiment.
        obs: raw CartPole observation (4,).
        Returns: (discrete_action, spike_count)
        """
        spikes = state_to_spikes(obs)
        spike_t = torch.tensor([spikes], dtype=torch.float32, device=self.device)
        logits, sc = self.forward(spike_t)
        action = int(logits.argmax(dim=1).item())
        return action, sc


# ---------------------------------------------------------------------------
# A2C Critic
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    Small MLP critic for A2C.

    Takes the raw 4-dim CartPole observation (position, velocity, angle,
    angular velocity) and outputs a scalar state-value estimate V(s).

    Kept separate from the SNN actor so its gradient path is clean ANN
    backprop — no surrogate-gradient complications.
    """

    def __init__(self, obs_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, obs_dim) -> value: (batch,)"""
        return self.net(obs).squeeze(-1)
