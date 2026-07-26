# Spiking Neural Networks for Adaptive Jerk-Limited Robot Motion 

Trying to develop a cerebellar-inspired spiking neural network that learns adaptive, low-jerk robot motion through spike-timing-based control. This repository contains a small spiking neural network (SNN) research sandbox:

- **MNIST sanity checks**: compare a dense ANN vs a simple LIF SNN.
- **CartPole controller benchmark**: compare a smooth analytical baseline (quintic planner) vs an SNN controller trained with surrogate gradients, with an optional STDP timing layer.
- **Evaluation framework**: log 4 metrics per episode and generate plots + a summary table from CSV outputs.

## Project layout

- `snn_research/`
  - `model.py`: ANN/SNN models (MNIST) + `QuinticPlanner` baseline + `SNNController` (SNN + optional STDP)
  - `data.py`: event/spike encoding for CartPole state (`state_to_spikes`, `encode_batch`)
  - `evaluate.py`: metric computation + FLOP estimation helpers
  - `train.py`: training loops and the 100-episode experiment runner (writes CSVs to `outputs/`)
  - `results.py`: plotting + summary table generation from the CSV outputs
  - `outputs/`: generated CSVs/figures (created automatically)
  - `notebooks/`: tutorials and smoke tests, plus a results analysis notebook

## Setup

You need Python and the packages used by the code (PyTorch, snntorch, gymnasium, torchvision, pandas, matplotlib, etc.).

If you already have an environment working, you can skip installs. Otherwise, a typical setup is:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install snntorch gymnasium pandas matplotlib numpy
python -m pip install nbformat nbconvert ipykernel
```

Notes:
- CartPole is run via `gymnasium`. If your environment complains about missing extras, you may need additional gymnasium dependencies depending on your setup.

## How to run (recommended)

Run modules from the **parent directory** (the folder that contains `snn_research/`), e.g.:

```bash
python -m snn_research.train --episodes 100 --train-episodes 200
```

This will:
- run the quintic baseline on CartPole
- train/evaluate the SNN controller variants (unless you pass `--skip-training`)
- write result CSVs into `snn_research/outputs/`

Then generate plots and a summary table:

```bash
python -m snn_research.results
```

Outputs are saved in `snn_research/outputs/` (PDF figures + `summary_table.csv`).

## Notebooks

The `snn_research/notebooks/` folder contains:

- **Tutorial notebooks**:
  - `01_lif_tutorial.ipynb`
  - `02_mnist_sanity.ipynb`
  - `03_cartpole_baseline.ipynb`
- **Smoke tests (quick “does this import/run?” notebooks)**:
  - `04_data_smoketest.ipynb`
  - `05_model_smoketest.ipynb`
  - `06_evaluate_smoketest.ipynb`
  - `07_train_import_smoketest.ipynb`
  - `08_results_smoketest.ipynb`
- **Results analysis**:
  - `09_results_analysis.ipynb` (loads CSVs from `outputs/`, regenerates figures/table, prints summary)

Tip: the smoke-test notebooks are intentionally lightweight and avoid running long training loops.

## Metrics (CartPole benchmark)

The benchmark logs four metrics per episode:

- **M1 peak_jerk**: maximum absolute 3rd derivative of cart position (smoothness proxy)
- **M2 endpoint_err**: final position error magnitude (accuracy proxy)
- **M3 spike_count / flops**: energy proxy (spikes for SNN, FLOP estimate as an upper bound)
- **M4 osc_amp**: RMS oscillation after the stop (vibration proxy)

## Troubleshooting

- **`ModuleNotFoundError: No module named 'snn_research'`**
  - Run from the parent directory using `python -m snn_research.<module>` (recommended), or ensure your working directory/PYTHONPATH includes the parent folder.

# Cerebellar-Inspired SNN Controllers for Jerk-Limited Robot Motion

**Research question:** *Can a biologically-plausible SNN — using LIF neurons and STDP-based timing — learn jerk-limited deceleration profiles that match or outperform polynomial trajectory planners, at lower computational cost?*

## Project Structure

```
snn_research/
├── data.py              Event encoding (continuous state → spike vectors)
├── model.py             SimpleANN, SimpleSNN, QuinticPlanner, SNNController, STDP
├── train.py             Training loops + 100-episode experiment runner
├── evaluate.py          4-metric benchmark (jerk, endpoint error, FLOPs, oscillation)
├── results.py           Plot generation and summary tables
├── notebooks/
│   ├── 01_lif_tutorial.ipynb       LIF neuron exploration & β sweeps
│   ├── 02_mnist_sanity.ipynb       SNN vs ANN on MNIST (sanity check)
│   └── 03_cartpole_baseline.ipynb  Quintic planner baseline on CartPole
└── outputs/             Saved models, figures, result CSVs
```

## Setup

Requires an NVIDIA GPU with CUDA support. Uses Conda for environment management.

```bash
# 1. Create environment
conda create -n snn_research python=3.11 -y
conda activate snn_research

# 2. Install PyTorch with CUDA (check nvidia-smi for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install research dependencies
pip install snntorch gymnasium matplotlib pandas numpy

# 4. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# 5. Verify full stack
python -c "import snntorch, gymnasium, torch; print('snntorch', snntorch.__version__); print('torch', torch.__version__)"
```

## Usage

### Exploration notebooks

Run from the `snn_research/notebooks/` directory:

```bash
cd snn_research/notebooks
jupyter notebook
```

1. **01_lif_tutorial.ipynb** — Start here. Explore LIF neuron dynamics.
2. **02_mnist_sanity.ipynb** — Train SNN on MNIST, verify >90% accuracy.
3. **03_cartpole_baseline.ipynb** — Establish quintic planner baseline numbers.

### Full experiment (100 episodes, all controllers)

```bash
python -m snn_research.train --episodes 100 --train-episodes 200
```

Options:
- `--episodes N` — evaluation episodes per controller (default: 100)
- `--train-episodes N` — training episodes for SNN controllers (default: 200)
- `--skip-training` — skip training, run evaluation with random weights

### Generate plots and summary

After running the experiment:

```bash
python -m snn_research.results
```

Produces figures in `snn_research/outputs/`:
- `fig1_jerk_comparison.pdf` — box plot of peak jerk
- `fig2_energy_tradeoff.pdf` — FLOPs vs jerk scatter
- `fig3_oscillation_comparison.pdf` — post-stop oscillation
- `fig4_endpoint_accuracy.pdf` — endpoint error
- `summary_table.csv` — mean ± std for all metrics (paper Table 1)

## Four Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **M1** Peak Jerk | max \|d³pos/dt³\| — smoothness of deceleration |
| **M2** Endpoint Error | \|final_pos - 0\| — accuracy of stopping |
| **M3** FLOPs / Spike Count | computational cost per inference step |
| **M4** Oscillation Amplitude | RMS position deviation after stop command |

## Controllers

1. **Quintic Polynomial Planner** — analytical baseline, minimises jerk from boundary conditions
2. **SNN (surrogate gradient only)** — LIF hidden layers trained with FastSigmoid surrogate
3. **SNN + STDP timing** — adds a Hebbian STDP layer for predictive deceleration onset

## References

- snnTorch documentation: https://snntorch.readthedocs.io
- Yamazaki & Tanaka (2007) — cerebellar forward model
- Casellato et al. (2014) — spiking cerebellar robot control
- Macfarlane & Croft (2003) — jerk-bounded trajectory planning
