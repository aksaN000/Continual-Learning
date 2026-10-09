# Continual Learning for Text Command Understanding

A PyTorch reference implementation comparing **Experience Replay** and **Elastic Weight Consolidation (EWC)** for mitigating catastrophic forgetting when a BERT-based intent classifier is trained sequentially across the domains of the HWU64 conversational dataset. The repository ships with a 16-configuration sweep over EWC strength and replay buffer size, plus the analysis pipeline that turns those runs into publication-grade plots and metrics.

**Paper:** [Continual Learning for Text Command Classification Using EWC and Experience Replay](paper.pdf) (Aksan Gony Alif, Nowmi Islam; BRAC University).

## Table of Contents

- [Background](#background)
- [What This Project Does](#what-this-project-does)
- [Results at a Glance](#results-at-a-glance)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Command Reference](#command-reference)
- [Experimental Design](#experimental-design)
- [Metrics](#metrics)
- [Configuration](#configuration)
- [Reproducing the Headline Numbers](#reproducing-the-headline-numbers)
- [Extending the Project](#extending-the-project)
- [References](#references)
- [License](#license)

## Background

Neural networks trained on a stream of tasks tend to overwrite parameters that were important for earlier tasks, a phenomenon called *catastrophic forgetting*. Two of the most influential mitigation strategies are:

- **Elastic Weight Consolidation (EWC)**, from Kirkpatrick et al. (2017), adds a quadratic penalty that pulls weights back toward values that were important (high Fisher information) for prior tasks.
- **Experience Replay** keeps a small buffer of examples from previous tasks and interleaves them with the current task's mini-batches.

This project implements both, evaluates them in isolation and in combination on a realistic conversational-NLU benchmark, and reports the practical trade-offs.

## What This Project Does

- Fine-tunes a `bert-base-uncased` classifier over the domains of HWU64 *sequentially*, one domain at a time.
- Implements **online EWC** with per-domain Fisher matrices, parameter masking at the 90th-percentile importance threshold, and an optional aggregation step for memory efficiency.
- Implements a configurable **replay buffer** with `balanced`, `importance`, and `diversity` sampling strategies.
- Tracks 12 continual-learning metrics per run (accuracy, F1, average/max forgetting, catastrophic events, backward and forward transfer, plasticity/stability, resource usage).
- Generates plots and a markdown summary across all 16 strategy configurations and produces a weighted ranking of the best trade-off.

## Results at a Glance

The repository ships with a completed 4 × 4 sweep over EWC λ ∈ {0, 1, 10, 50} and replay buffer size ∈ {0, 100, 500, 1000}, run on the HWU64 sequence. The full per-strategy table lives at [comprehensive_analysis/metrics_summary.md](comprehensive_analysis/metrics_summary.md).

**Top 5 configurations by average accuracy across all domains:**

| Rank | EWC λ | Replay size | Avg. accuracy | Avg. F1 | Avg. forgetting | Max forgetting | Catastrophic events |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 500  | **0.8283** | 0.8097 | 0.0 | 0.258 | 1 |
| 2 | 10 | 500  | 0.8278     | 0.8091 | 0.0 | 0.260 | 1 |
| 3 | 1  | 500  | 0.8273     | 0.8085 | 0.0 | 0.262 | 1 |
| 4 | 50 | 0    | 0.8260     | 0.8063 | 0.0 | 0.251 | 0 |
| 5 | 10 | 0    | 0.8255     | 0.8058 | 0.0 | 0.254 | 0 |

**Selected by the weighted composite score** (balancing accuracy, forgetting, transfer, plasticity/stability, and resource cost; see [comprehensive_analysis/best_config_explanation.txt](comprehensive_analysis/best_config_explanation.txt)):

- **Tied at 3.16:** `EWC=0, replay=1000` and `EWC=0, replay=0`
- **Tied at 3.15:** all four `replay=500` configurations and the three pure-EWC configurations

Practical takeaways from the sweep:

1. **A 500-example replay buffer + non-trivial EWC λ gives the best raw accuracy** (~83%), regardless of whether λ is 1, 10, or 50: the accuracy ceiling is largely set by replay, with EWC strength being secondary.
2. **Pure EWC (no replay) avoids catastrophic-forgetting events entirely** at the cost of ~0.2 absolute accuracy points versus the best combined configuration.
3. **Maximum forgetting is bounded around 0.25–0.32 across all configurations**, including the baseline: BERT's pre-trained representations naturally limit how badly the model can collapse on prior domains.
4. **More replay does not strictly help.** The 1000-example buffer is essentially tied with the 500-example buffer; the 100-example buffer is the worst replay setting on accuracy.
5. **EWC overhead is modest:** the ewc_overhead_ratio column shows fewer than 13% additional ops for the heaviest EWC setting.

Full visualisations (radar comparison, plasticity/stability scatter, forgetting curves, per-strategy resource bars) are pre-generated under [comprehensive_analysis/](comprehensive_analysis/).

## Repository Layout

```
.
├── data/
│   ├── download_data.py        # Fetches HWU64 from xliuhw/NLU-Evaluation-Data and organises it by domain
│   └── data_utils.py           # CommandDataset, tokenisation helpers, per-domain DataLoaders
├── models/
│   ├── base_model.py           # TextCommandClassifier: BERT + dropout + linear head
│   └── continual_learner.py    # ContinualTextCommandLearner: online EWC, Fisher info, parameter masking
├── training/
│   ├── train.py                # Per-domain training loop and evaluation
│   ├── replay_buffer.py        # Replay buffer with balanced / importance / diversity sampling
│   └── metrics.py              # 12 continual-learning metrics and aggregations
├── experiments/
│   ├── run_experiment.py       # Single-experiment driver, results JSON dump
│   └── configs/default_config.py
├── visualization/
│   └── plot_results.py         # Per-experiment plots, strategy comparison plots
├── enhanced_visualization.py   # Builds the comprehensive_analysis report
├── full_batch_results/         # 16 completed runs from the 4 × 4 sweep + comparison plots
├── comprehensive_analysis/     # Cross-strategy figures, metrics_summary.md, weighted ranking
├── main.py                     # CLI entry point (prepare-data, run-experiment, batch, visualize, compare, analyze)
├── FulllGuide.txt              # Long-form walkthrough including PowerShell commands
├── sample_report.txt           # Earlier results write-up with placeholder figures
├── relatedPapers.txt           # Annotated citations for EWC, replay, BERT, and CL metrics
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or newer.
- A CUDA-capable GPU is strongly recommended. The sweep was originally run on consumer hardware; a single full-batch experiment takes a few minutes to a few hours depending on GPU and epoch count.
- ~1.5 GB free disk for the HWU64 raw dataset and processed splits.

### Installation

```bash
git clone https://github.com/aksaN000/Continual-Learning.git
cd Continual-Learning

# Create and activate a virtual environment
python -m venv cl_env
.\cl_env\Scripts\Activate.ps1     # Windows PowerShell
# source cl_env/bin/activate      # Linux / macOS

# PyTorch with CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Project dependencies
pip install -r requirements.txt
```

### Prepare the dataset

```bash
python main.py prepare-data
```

This downloads HWU64 from [xliuhw/NLU-Evaluation-Data](https://github.com/xliuhw/NLU-Evaluation-Data), splits the annotations by domain (e.g. `alarm`, `audio`, `calendar`, `iot`, …), and writes per-domain train/test splits into `data/processed/`.

## Command Reference

`main.py` exposes the full pipeline through `argparse` subcommands.

| Command | Purpose | Key flags |
| --- | --- | --- |
| `prepare-data` | Download and process HWU64 | none |
| `run-experiment` | Run a single sequential training pass | `--name`, `--use_ewc` / `--no_ewc`, `--use_replay` / `--no_replay`, `--ewc_lambda`, `--replay_buffer_size`, `--replay_batch_size`, `--epochs`, `--learning_rate`, `--seed`, `--config` |
| `batch` | Run a full grid of EWC λ × replay-size configurations | `--ewc_values 0 1 10 50`, `--replay_sizes 0 100 500 1000`, `--base_config`, `--output_dir` |
| `visualize` | Render plots for a single results JSON | `--results`, `--output_dir` |
| `compare` | Render cross-strategy comparison plots from a results directory | `--results_dir`, `--output_dir`, `--strategies` |
| `analyze` | Build the full multi-figure analysis report (radar, ranking, forgetting, resource, …) | `--results_dir`, `--output_dir` |

### Example: minimal sanity check

```bash
python main.py run-experiment --name smoke --use_ewc --use_replay --epochs 1
```

### Example: ablation across strategies

```bash
python main.py run-experiment --name baseline_naive --no_ewc  --no_replay
python main.py run-experiment --name ewc_only       --use_ewc --no_replay
python main.py run-experiment --name replay_only    --no_ewc  --use_replay
python main.py run-experiment --name combined       --use_ewc --use_replay
python main.py compare --results_dir results/
```

### Example: full sweep (reproduces `full_batch_results/`)

```bash
python main.py batch --ewc_values 0 1 10 50 --replay_sizes 0 100 500 1000
python main.py analyze --results_dir full_batch_results --output_dir comprehensive_analysis
```

## Experimental Design

### Dataset

[HWU64](https://github.com/xliuhw/NLU-Evaluation-Data) is a multi-domain natural-language-understanding benchmark from Heriot-Watt University covering 64 intents across personal-assistant scenarios such as alarms, calendar, audio, IoT, weather, music, and Q&A. The repository organises it by scenario into per-domain directories under `data/processed/`, treating each scenario as a separate continual-learning task.

### Model

`models/base_model.py` defines a thin classifier on top of `bert-base-uncased`:

- BERT-base encoder (frozen tokenisation, full fine-tuning of weights)
- Dropout (`p = 0.1`)
- Linear classification head spanning all domain labels concatenated

`models/continual_learner.py` wraps the classifier with continual-learning state:

- Per-domain label ranges and `get_domain_logits` slicing
- Per-domain Fisher information matrices (`update_ewc_params`)
- Optional aggregated importance via `consolidate_ewc_online`
- 90th-percentile parameter mask that focuses the EWC penalty on the most important weights

### Default hyperparameters

Defined in [experiments/configs/default_config.py](experiments/configs/default_config.py):

| Group | Setting | Value |
| --- | --- | --- |
| Data | `max_length` | 128 |
| Data | `batch_size` | 16 |
| Model | `base_model` | `bert-base-uncased` |
| Model | `dropout` | 0.1 |
| Training | `epochs` (per domain) | 3 |
| Training | `learning_rate` | 1e-5 |
| Training | `weight_decay` | 0.01 |
| Continual | `replay_buffer_size` | 750 |
| Continual | `replay_batch_size` | 10 |
| Continual | `ewc_lambda` | 10.0 |
| Continual | `replay_strategy` | `balanced` |
| Experiment | `online_ewc` | `True` |
| Experiment | `importance_threshold` | 85 |
| Experiment | `seed` | 42 |

## Metrics

Each experiment dumps a `results.json` containing the following metric families (computed in `training/metrics.py` and surfaced in `comprehensive_analysis/`):

| Family | Metric | What it measures |
| --- | --- | --- |
| Performance | `avg_accuracy` | Mean accuracy across all seen domains at the end of training |
| Performance | `avg_f1_score` | Macro-F1 averaged across all seen domains |
| Forgetting  | `avg_forgetting` | Mean drop in per-domain accuracy from peak to final |
| Forgetting  | `max_forgetting` | Worst-case per-domain drop |
| Forgetting  | `catastrophic_forgetting_events` | Count of drops larger than a configurable threshold |
| Transfer    | `avg_backward_transfer` | How learning later domains affects earlier ones |
| Transfer    | `avg_forward_transfer` | How prior training improves later domain accuracy |
| Plasticity-Stability | `plasticity` | Capacity to acquire new knowledge |
| Plasticity-Stability | `stability` | Capacity to retain prior knowledge |
| Plasticity-Stability | `plasticity_stability_ratio` | Trade-off ratio between the two |
| Resource    | `buffer_utilization` | Fraction of replay buffer actually used |
| Resource    | `ewc_overhead_ratio` | Compute overhead of the EWC penalty term |

## Configuration

For more involved sweeps, copy `experiments/configs/default_config.py` and pass it to either `run-experiment` (`--config path/to/config.py`) or `batch` (`--base_config path/to/config.json`). The CLI flags listed in the [command reference](#command-reference) override any value provided in a config file.

Replay sampling strategy can be switched via the `continual.replay_strategy` key:

- `balanced`: equal samples per stored domain (default)
- `importance`: bias toward examples the model previously got wrong
- `diversity`: bias toward examples with high embedding-space spread

## Reproducing the Headline Numbers

The shipped figures and tables in `comprehensive_analysis/` correspond exactly to the 16 runs in `full_batch_results/`. To regenerate them end-to-end:

```bash
python main.py prepare-data
python main.py batch --ewc_values 0 1 10 50 --replay_sizes 0 100 500 1000 --output_dir full_batch_results
python main.py analyze --results_dir full_batch_results --output_dir comprehensive_analysis
```

Per-experiment plots can be regenerated with:

```bash
python main.py visualize --results full_batch_results/ewc10.0_replay500_<timestamp>/results.json --output_dir <out>
```

## Extending the Project

Natural directions for further work, many of them noted in [future_expansions](future_expansions):

- **More CL methods:** generative replay, Learning without Forgetting (LwF), Progressive Neural Networks, dynamic architecture expansion.
- **Alternative backbones:** swap `bert-base-uncased` for `roberta-base`, `distilbert-base-uncased`, or `albert-base-v2` in `models/base_model.py`.
- **Smarter replay:** importance-weighted, gradient-similarity, or coreset-based example selection.
- **Harder domain sequences:** randomise or adversarially order the HWU64 scenarios to widen the forgetting gap and make the comparison more discriminating.
- **Statistical testing:** add paired bootstrap or Wilcoxon tests over multiple seeds, since the current sweep is single-seed.

## References

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS 114(13), 3521–3526. doi:10.1073/pnas.1611835114
2. Lopez-Paz, D., & Ranzato, M. (2017). *Gradient Episodic Memory for Continual Learning.* NeurIPS 30.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT, 4171–4186.
4. Maltoni, D., & Lomonaco, V. (2019). *Continuous learning in single-incremental-task scenarios.* Neural Networks 116, 56–73.
5. Kemker, R., McClure, M., Abitino, A., Hayes, T. L., & Kanan, C. (2018). *Measuring Catastrophic Forgetting in Neural Networks.* AAAI-18.
6. Liu, X., Eshghi, A., Swietojanski, P., & Rieser, V. (2019). *Benchmarking Natural Language Understanding Services for Building Conversational Agents.* (HWU64 dataset.)

## License

Released under the MIT License. See `LICENSE` if present, otherwise the standard MIT terms apply.

## Citation

If you use this code, please cite as:

```bibtex
@misc{continual_learning_text_commands,
  title  = {Continual Learning for Text Command Understanding},
  author = {aksaN000},
  year   = {2025},
  howpublished = {\url{https://github.com/aksaN000/Continual-Learning}}
}
```
