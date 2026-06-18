## Cerebellar-Inspired SNN Controllers for Jerk-Limited Robot Motion: An Energy-Efficient Alternative to Polynomial Trajectory Planning

This is the full blueprint and the complete written plan. Everything I need to actually execute this.

---

## The Research Question (my anchor)

> *"Can a biologically-plausible Spiking Neural Network — using LIF neurons and STDP-based timing — learn jerk-limited deceleration profiles that match or outperform polynomial trajectory planners, at lower computational cost?"*

Every decision in this project traces back to this question. If something doesn't help answer it, cut it.

---

## Phase 1 — Foundation (Weeks 1–3)

**What I'm doing:** Building conceptual fluency in three areas simultaneously so I'm not blocked when coding starts.

**Track A: SNN mechanics**
- Install snnTorch: `pip install snntorch torch gymnasium matplotlib`
- Run their LIF neuron tutorial end-to-end. Don't just read it — change the τ (time constant) and β (decay) values and observe what happens to the spike pattern. This intuition is critical for Phase 3.
- Read: *"Spiking Neural Networks: The Future of Brain-Inspired Computing"* (arXiv 2510.27379) — focus on Section 3 (energy metrics) and Section 5 (training strategies). I don't need the hardware sections deeply yet.

**Track B: Cerebellar control literature**
- The cerebellum is a spiking network responsible specifically for smooth motion timing. This is my biological justification.
- Key papers: Yamazaki & Tanaka (2007) on cerebellar forward models, and Casellato et al. (2014) on cerebellar-inspired robot control. Both are free on Google Scholar.
- I don't need to implement their models — I need to understand the *principle*: the cerebellum fires predictive corrective signals *before* error occurs. my STDP layer will do this computationally.

**Track C: Current engineering baseline**
- Read the Wikipedia-level math on S-curves and quintic polynomial trajectory planning. I'm implementing these as my comparison baseline in Phase 2, so I need to understand the jerk formula: jerk = d³position/dt³.
- The key insight: polynomial planners are hand-engineered. They minimize jerk analytically. I'm replacing that with a network that *learns* to minimize jerk.

**Week 3 checkpoint:** I should be able to run a 2-layer snnTorch feedforward network on MNIST and get >90% accuracy. If I can do that, my environment is correct.

---

## Phase 2 — Build & Validate (Weeks 4–8)

This phase has two parallel tracks and one deliberate throwaway.

**The throwaway (Weeks 4–5, ~5 days max): MNIST sanity check**

Build ANN vs SNN on MNIST. Measure accuracy, FLOPs, and spike sparsity. This is *not* my research — it's proof my implementation is correct. It becomes Appendix A in my paper, 2 paragraphs. Don't spend more than a week here.

Expected result: SNN gets ~96% vs ANN's ~98%, but uses 85–90% fewer FLOPs. This validates my energy proxy approach before I move to robotics.

**The baseline (Weeks 5–7): Polynomial planner on cart-pole**

```python
# my file structure
data.py       # event encoding from Gymnasium state
model.py      # ANN, SNN (LIF), polynomial planner
train.py      # training loop, spike/energy logging
evaluate.py   # 4-metric benchmark
results.py    # plots and tables
```

Set up `gymnasium` cart-pole. Implement a quintic polynomial planner that smoothly decelerates the cart to a stop. Log these four metrics for *every* controller I test:
1. Peak jerk (∂³pos/∂t³) — the core metric
2. Endpoint positional error (accuracy proxy)
3. FLOPs or spike count per inference step (energy proxy)
4. Oscillation amplitude after stop (vibration proxy — measure how long the pole rings after stopping)

This gives I clean baseline numbers. my SNN must beat the polynomial planner on metrics 1, 3, and 4 — and come close on metric 2. That's my result.

**The SNN controller (Weeks 6–8, overlapping): LIF + STDP**

Architecture:
- Input layer: event-encoded state (velocity crossings → ON/OFF spikes)
- Hidden layers: 2–3 LIF layers with surrogate gradient backprop for initial training
- STDP timing layer: learns *when* to fire deceleration commands based on trajectory history
- Output layer: torque/force command decoded from spike rate

The STDP layer is my novel contribution. It's not trained with gradient descent — it updates weights based on spike co-occurrence timing. This is biologically plausible and computationally cheap. It's what allows my network to "predict" when to decelerate, mimicking how the cerebellum fires ahead of error.

Event encoding (important — this is how I bridge robotics to spikes):
```python
# Convert continuous cart velocity to spike events
def velocity_to_spikes(v, threshold=0.1):
    if v > threshold:  return [1, 0]  # ON event
    elif v < -threshold: return [0, 1]  # OFF event
    else:              return [0, 0]  # no event
```

---

## Phase 3 — Robotics Experiment (Weeks 9–11)

**What I'm building:** The same cart-pole task, but now my SNN controller is doing the motion planning instead of the polynomial planner. I'm measuring whether the SNN produces smoother (lower jerk) stops.

**Experiment design:**

Run 100 episodes per controller. Log all four metrics per episode. my results table will look like:

| Controller | Peak Jerk | Endpoint Error | FLOPs/step | Osc. After Stop |
|---|---|---|---|---|
| Quintic poly | baseline | baseline | baseline | baseline |
| SNN (surrogate grad only) | ? | ? | -70–80% | ? |
| SNN + STDP timing | ? | ? | -75–85% | ? |

I want to show that the STDP version reduces jerk *and* oscillation compared to the surrogate-only SNN, while maintaining similar accuracy — this validates that learned timing (STDP) adds value on top of the basic energy savings.

**If the SNN underperforms on jerk:** This is still a valid result. A paper saying "SNNs save energy but don't yet match polynomial planners on jerk — here's why and here's what's needed" is honest, publishable, and arguably more useful than a paper that cherry-picks positive results. Don't fear this outcome.

**Optional level-up (if time allows, Week 11):** Swap cart-pole for a 2-DOF robot arm in PyBullet or MuJoCo (both free, laptop-friendly). The physics are more realistic and the jerk problem is more visible. But don't do this at the cost of finishing my metrics.

---

## Phase 4 — Analysis & Write-Up (Weeks 12–14)

**Paper structure (my exact outline):**

1. **Abstract** — 150 words. State the problem (jerk in robot motion), my approach (SNN + STDP), and my key result (X% jerk reduction at Y% lower compute).

2. **Introduction** — Energy crisis in robotics AI, why spiking is biologically motivated, why trajectory smoothing is the right test task. End with my research question verbatim.

3. **Related Work** — Three buckets: (a) SNN energy efficiency (SpikeLLM, SpikeMamba, my arXiv refs), (b) Jerk-limited trajectory planning (S-curves, polynomial methods), (c) Cerebellar-inspired controllers (Yamazaki, Casellato). Note the gap: no prior work combines SNN controllers specifically with jerk minimization as the evaluation criterion.

4. **Method** — LIF neuron math, STDP update rule, event encoding scheme, training procedure. Include my architecture diagram.

5. **Experiments** — Cart-pole setup, all four metrics, 100-episode averages with standard deviation. my table from Phase 3.

6. **Discussion** — What the results mean, why STDP helps (or doesn't), limitations (noisy spikes, sim-to-real gap, GPU scaling), future work (real hardware, DVS camera input).

7. **Conclusion** — One paragraph.

**Target venues (in order of fit):**
- ICRA 2027 workshop on neuromorphic robotics (abstract deadline usually Sept–Oct 2026)
- IROS 2026 late-breaking results
- arXiv preprint first — post it before the deadline, it establishes priority and gets my work visible

---

## Hardware & Setup — my Laptop Is Enough

Yes, this is entirely software. Here's the breakdown:

**I need:**
- Python 3.10+
- `snnTorch` — runs on CPU fine for cart-pole scale
- `gymnasium` — pure Python simulation, no GPU
- `numpy`, `matplotlib`, `pandas` — standard
- `torch` — CPU mode is sufficient; snnTorch's MNIST tutorial runs in under 10 minutes on CPU

**I don't need:**
- A GPU (snnTorch on CPU is fast enough for cart-pole; MNIST takes ~10 min on CPU, cart-pole is faster)
- Cloud compute (AWS, Colab not needed — though free Colab T4 is a nice backup)
- Physical hardware, sensors, or a robot
- Any paid software or licenses

**The one caveat:** If I upgrade to MuJoCo/PyBullet in Phase 3, those simulators have slightly heavier CPU load but still run fine on a modern laptop. Cart-pole in Gymnasium is essentially instantaneous.

**Recommended setup:**
```bash
pip install snntorch torch gymnasium matplotlib pandas numpy
# Test immediately:
python -c "import snntorch, gymnasium; print('All good')"
```

Use VS Code + Jupyter notebooks for experimentation, then clean Python scripts for the final reproducible version I push to GitHub.

---

## The Honest Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| SNN doesn't beat poly planner on jerk | Medium | Reframe as "SNN matches jerk at 80% lower compute" — still novel |
| STDP layer is hard to tune | Medium-High | Fall back to surrogate-only SNN; STDP becomes "future work" |
| Cart-pole too simple to show jerk clearly | Low | Use velocity-limited stop task instead of full cart-pole |
| Results not statistically clean | Low | 100 episodes + standard deviation handles this |

The biggest real risk is STDP tuning — the learning window parameters (A+, A−, τ+, τ−) need careful selection. Budget Week 8 as a debugging week specifically for this. snnTorch's documentation has STDP examples.

---

## What Makes This Master's-Level Work

The combination that makes this my contribution and not just a replication study:

1. **Framing** — I'm the first to explicitly test SNN controllers against S-curve planners using *jerk* as the primary metric. Prior SNN-robotics work measures accuracy or energy, not smoothness.

2. **STDP as timing mechanism** — Using STDP specifically for the deceleration onset prediction problem is novel at student scale and not done in prior cart-pole literature.

3. **The four-metric benchmark** — Jerk + accuracy + energy + oscillation together is a more complete evaluation than any individual prior student project.

4. **The honest null result** — If SNNs don't fully close the gap, my paper explains *why* (sparsity requirements, discrete timesteps vs continuous polynomial math) — that's a contribution to the field, not a failure.

This is solid, defensible, laptop-runnable Master's research. Start with the snnTorch LIF tutorial this week.

Here's your complete research manual — an interactive HTML document covering every step of the 14 weeks.

**What's inside:**

- **Prerequisites & Setup** — exact pip commands, file structure to create on day one, and an immediate verification test
- **Phase 1 (Wks 1–3)** — three parallel learning tracks, a concept table for LIF/STDP/surrogate gradients, the event encoding function to implement, and a clear Week 3 checkpoint
- **Phase 2 (Wks 4–8)** — the MNIST throwaway (with a hard "stop after 5 days" warning), the four-metric logging framework with full code, the SNN+STDP architecture explained layer by layer, and the STDP weight update rule
- **Phase 3 (Wks 9–11)** — the 100-episode runner code, your results table template, and explicit framing for both positive and negative outcomes
- **Phase 4 (Wks 12–14)** — matplotlib figure code for the two key plots, submission venues in priority order
- **Paper Guide** — expandable section-by-section writing guide with word counts, formulas, and pitfalls
- **Risk Table** — every real risk with concrete mitigations
- **Math Reference** — all 8 equations you need to know cold, with plain-English explanations

The single most important thing to do today: run the pip install block and get the Week 3 checkpoint (snnTorch + MNIST + >90% accuracy) before touching anything else.