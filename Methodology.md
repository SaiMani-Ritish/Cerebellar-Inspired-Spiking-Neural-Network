This is a good question to resolve clearly before you write a single line of training code, because the framing of "dataset" is the first place a reviewer — or your professor — could push back. Let me walk through all three areas in full.

---

## On the "dataset" question

This project does not use a static dataset in the supervised learning sense. What you have instead is a **dynamical system generating trajectory data on-demand** — the Gymnasium CartPole-v1 simulator. Each episode is a sequence of (state, action, next-state) tuples produced by running a controller inside a physics engine. The "data" is the position trace over time.

This distinction matters for your narrative. You are not training a classifier on labeled examples. You are training a controller that interacts with an environment and produces trajectories, and you are evaluating the *shape* of those trajectories — specifically their third derivative. Your comparison is between the trajectory shapes produced by three different controllers under identical initial conditions.

The honest framing for your paper is: **the dataset is a controlled experimental protocol, not a corpus**. You fix random seeds (`seed=episode_number` for reproducibility), fix initial conditions across controllers, and run 100 independent episodes per controller. The resulting position arrays are your data. Because all three controllers see the same initial states in the same order, differences in the resulting jerk profiles are attributable to the controller, not to sampling variation.

One important clarification about cart-pole specifically: the standard CartPole-v1 task is a balancing task where the episode terminates when the pole falls. That is not a stopping task — it does not produce a deceleration event for you to measure jerk on. You need to modify the task to create a **velocity-limited stop task**: the controller must decelerate the cart from some initial velocity `v₀` to `v=0` at position `x=0`. You can implement this either by building a custom Gymnasium wrapper that resets with a nonzero cart velocity and rewards stopping behavior, or by using the cart position trace from a natural CartPole episode and treating the final deceleration segment as your measurement window. The former is cleaner and more defensible; the latter is faster to prototype.

---

## Training procedure in full

There are three separate learning systems in your architecture, and they are trained in two sequential phases with different objectives. Conflating them is the most common conceptual error in SNN papers, so let's be precise.

### Phase A — Surrogate gradient training of the LIF hidden layers

The LIF hidden layers (2–3 layers depending on what you settle on) are trained end-to-end with backpropagation through time (BPTT), using a surrogate gradient to handle the non-differentiability of the spike function.

The spike function is a step: `s = 1 if V > V_th, else 0`. Its true gradient is zero almost everywhere and undefined at threshold. The surrogate gradient replaces this with a smooth approximation — snnTorch uses the `FastSigmoid` surrogate by default:

```
∂s/∂V ≈ 1 / (1 + |V - V_th| · k)²
```

where `k` controls the sharpness of the approximation. The forward pass uses the true step function (producing real binary spikes); the backward pass substitutes this smooth approximation. This is a deliberate mathematical inconsistency that works in practice because the smooth gradient points in approximately the right direction.

Your training target in this phase is a **behavioral objective**, not jerk directly. You train the hidden layers to produce deceleration-appropriate motor commands — meaning you reward the controller for reducing cart velocity smoothly toward zero, penalizing both positional overshoot and large control forces. A simple reward like `r = -|v| - α·|F|` (penalize residual velocity and force magnitude) is sufficient. You are not yet asking the network to minimize jerk — you are asking it to stop smoothly. Jerk reduction emerges from force continuity; you measure it but do not directly optimize it during training.

Concretely:

- Unroll the SNN for `num_steps=25` timesteps per control decision
- At each environment step, feed the 8-dimensional spike-encoded state into the network
- The output layer produces a spike-rate-decoded force command
- Accumulate the BPTT gradient across the 25 internal timesteps
- Update hidden layer weights with Adam
- Repeat across episodes until the controller reliably decelerates to stop

The STDP layer weights are **frozen** during this phase. Initialize them with small random values and do not update them. This ensures the hidden layers learn to produce interpretable pre-synaptic spike patterns for the STDP layer to work with.

### Phase B — STDP online updates of the timing layer

Once the hidden layers are trained and frozen, you switch to online STDP updates on the timing layer alone. The STDP rule is:

```
ΔW_ij = A+ · exp(−Δt/τ+)    if pre_i fires before post_j  (LTP)
ΔW_ij = −A− · exp(Δt/τ−)   if post_j fires before pre_i  (LTD)
```

where `Δt = t_post − t_pre`. The rule runs during episode execution — after each control step, you compute the pairwise spike-timing relationships between the STDP layer's inputs (hidden layer outputs) and its outputs (deceleration commands), and apply the weight update immediately. This is online, unsupervised learning in the strict sense: no gradient, no loss function, no optimizer step.

What the STDP layer is learning, concretely: the hidden layers will have learned to fire certain patterns when the cart is at velocity `v` with `T` timesteps until the target stop position. The STDP layer observes these patterns and, over many episodes, strengthens the synapses that connect "velocity-crossing spike at time t" to "deceleration command spike at time t + Δt_optimal." It is learning the temporal offset between observable event and required response. Episodes where the timing was good (smooth stop, low jerk) reinforce the corresponding spike-timing associations; episodes where timing was off produce weak or negative updates.

Key hyperparameters you will need to tune:
- `A+` and `A−`: learning rates for potentiation and depression. Start equal (`0.01` each). If STDP diverges (weights grow unbounded), add a soft weight bound: `W = W_max · tanh(W / W_max)`.
- `τ+` and `τ−`: temporal windows in milliseconds. `20ms` each is biologically standard. If your simulation timestep is `dt=0.02s`, then `20ms` = 1 timestep — that is very narrow. You may need to scale your τ values to match your simulation's temporal resolution. Try `τ = 5` timesteps as a starting point.
- Weight initialization: small positive values (`0.01` range). STDP is sensitive to initialization because it has no error signal to pull it out of bad local structures.

The two phases can run sequentially (train hidden layers for N episodes, then freeze and run STDP for M episodes) or in a staged alternating schedule. Sequential is easier to debug and reason about; use it first.

---

## Metrics and the honest narrative

Here is where your framing has to be careful, because you are making an argument that an SNN *can* learn timing, not that it *optimally* solves a trajectory planning problem. The metrics need to support a narrative about the *nature* of the controller's behavior, not just its score on a leaderboard.

### Peak jerk — your primary metric, but measure its distribution

`J_peak = max|d³x/dt³|` across the episode. Numerically this is three consecutive `np.gradient` calls on the position array. The important thing is to report **the distribution across 100 episodes** (mean ± std, plus a box plot), not just the mean. Here is why this is important for your narrative: the polynomial planner is deterministic — it produces the same jerk profile every episode for the same initial condition. Your SNN is stochastic — spike timing has variability. The honest story is not "SNN achieves lower mean jerk" but "SNN achieves comparable mean jerk with different variance characteristics." If the SNN's jerk distribution has heavier tails than the polynomial (which is likely), that tells you something real about the cost of adaptivity — the same stochasticity that allows online learning also introduces variability in timing. That is a substantive finding, not a failure.

Also measure **jerk at two specific moments** — deceleration onset and stop — rather than just the episode peak. This lets you separate timing errors from force-control errors, which maps onto your mechanistic story about STDP learning onset timing.

### Endpoint positional error — the validity check

`E_pos = |x_final − x_target|`. This tells you whether the controller is actually completing the task. If your SNN has dramatically lower endpoint error than the polynomial planner, something is wrong (the planner should be analytically optimal here). If the SNN has dramatically higher error, your jerk comparison is unfair — you are comparing a controller that stopped correctly against one that did not stop correctly. The honest narrative requires that both controllers are actually solving the same task. Report this metric as a **validity constraint**, not a competition: the comparison is only meaningful when endpoint error is in the same ballpark across controllers.

### FLOPs per step — be precise about what you are counting

The standard proxy is: one spike = one synaptic operation (MAC). An ANN layer with `N_in × N_out` weights performs `N_in × N_out` multiply-accumulate operations per forward pass, always. An SNN layer performs `spike_count × N_out` operations per forward pass, where `spike_count` is the number of active pre-synaptic neurons — and at 10–15% firing rate, this is roughly `0.1 × N_in × N_out`. The ratio is your efficiency claim.

But be honest about what this does and does not mean. This is a **theoretical energy proxy**, not a measured power draw. You are counting operations, not joules. On a standard GPU, the SNN may actually be *slower* than the ANN because sparse operations are not well-vectorized on GPU hardware (neuromorphic chips like Intel Loihi exploit sparsity in hardware; GPUs do not). Your claim is about operation count, which is the relevant metric for neuromorphic deployment — make that scope explicit.

### Post-stop oscillation amplitude — your most novel metric

`A_osc = RMS(x[t_stop : t_stop + window] − x_target)` over a 50-step window after stopping. This measures how much the cart rings after the control episode ends. The polynomial planner, being open-loop, stops at the mathematically computed endpoint — but if any perturbation shifted the trajectory, the cart will overshoot and settle. The SNN, if STDP is working correctly, should produce a more gradual deceleration that dissipates momentum progressively rather than terminating abruptly. This maps directly onto the jerk story: smooth deceleration onset = low jerk = low residual momentum at stop = low oscillation.

This metric is also the one most sensitive to whether STDP is actually contributing. Compare oscillation amplitude between the surrogate-only SNN and the STDP SNN. If STDP is doing real work on timing, oscillation should drop. If it is not, oscillation is similar and you have evidence that STDP did not successfully learn timing associations — which is itself a finding worth analyzing.

---

## The honest narrative structure

Given the above, here is what your results section should actually argue, regardless of which direction the numbers go:

**If STDP reduces jerk and oscillation relative to surrogate-only:** The narrative is "STDP learns a causal timing association between velocity spikes and deceleration onset, producing the smooth deceleration curve predicted by the §4 mechanistic argument, at ~80% lower operational cost than the polynomial baseline." This is the strong version of your hypothesis.

**If STDP matches but does not beat surrogate-only on jerk:** The narrative is "STDP regularizes temporal variability without improving mean jerk, suggesting that the hidden layers already learn sufficient timing structure through surrogate gradient training, and STDP provides secondary stabilization. The energy efficiency claim holds regardless." This is a legitimate and honest intermediate result.

**If the SNN underperforms the polynomial on jerk:** The narrative is "Discrete timestep quantization in the SNN (dt = 20ms) creates a temporal resolution floor that prevents the STDP layer from learning timing offsets finer than one simulation step — whereas the polynomial planner solves a continuous differential equation. The energy savings (70–80% FLOPs/step) are confirmed, but matching polynomial smoothness requires either finer temporal resolution or an adaptive threshold mechanism. We characterize exactly where the gap comes from." This is a falsified hypothesis reported honestly, and it is publishable because it explains *why* the gap exists with mechanical precision — that explanation is a contribution.

The master's-level quality of this work is in the richness of the mechanistic story, not in achieving the best jerk number. Your hypothesis is falsifiable, your metrics discriminate between competing explanations, and your experimental design controls for confounds. That is what makes it research.