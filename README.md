# Continual Learning for Text Command Understanding

A PyTorch implementation of continual learning techniques to prevent catastrophic forgetting in sequential text command understanding.

## Overview

This project implements a continual learning system that can sequentially learn to understand different categories of text commands without forgetting previously learned knowledge. The implementation focuses on:

1. **Experience Replay**: Storing and revisiting examples from previous domains
2. **Elastic Weight Consolidation (EWC)**: Regularizing important model parameters to prevent forgetting

## Features

- BERT-based text classification model
- Replay buffer for experience replay
- EWC regularization to prevent catastrophic forgetting
- Comprehensive metrics tracking and visualization
- Experiment configuration system
- Batch experimentation capabilities

## Project Structure

```
continual_learning_project/
├── data/
│   ├── download_data.py        # Script to download and prepare datasets
│   └── data_utils.py           # Helper functions for data processing
├── models/
│   ├── base_model.py           # Base model definition (BERT adaptation)
│   └── continual_learner.py    # Continual learning implementation
├── training/
│   ├── train.py                # Main training loop
│   ├── replay_buffer.py        # Memory replay implementation
│   └── metrics.py              # Evaluation metrics
├── experiments/
│   ├── run_experiment.py       # Script to run complete experiment pipeline
│   └── configs/                # Configuration files for experiments
├── visualization/
│   └── plot_results.py         # Visualization of forgetting metrics
├── main.py                     # Main entry point
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/continual-learning-project.git
cd continual-learning-project
```

2. Create and activate a virtual environment:

```bash
# For Windows
python -m venv cl_env
.\cl_env\Scripts\activate

# For Unix/MacOS
python -m venv cl_env
source cl_env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Preparing the Dataset

```bash
python main.py prepare-data
```

This will download and process the HWU64 dataset, organizing it by domains for sequential learning.

### Running a Single Experiment

```bash
python main.py run-experiment --name my_experiment --use_ewc --use_replay
```

This will run a continual learning experiment with both EWC regularization and experience replay enabled.

### Running Multiple Experiments

```bash
python main.py batch --ewc_values 0 1 5 --replay_sizes 0 500 1000
```

This will run experiments with different combinations of EWC strength and replay buffer sizes.

### Visualizing Results

```bash
python main.py visualize --results results/my_experiment_20240414_123456/results.json --output_dir visualizations
```

This will create visualizations for a single experiment.

### Comparing Strategies

```bash
python main.py compare --results_dir results/ --output_dir comparisons
```

This will compare the performance of different continual learning strategies.

## Experiment Configuration

You can customize experiments using command-line arguments or configuration files. Key parameters include:

- `--replay_buffer_size`: Size of the experience replay buffer
- `--ewc_lambda`: Strength of EWC regularization
- `--epochs`: Number of epochs to train on each domain
- `--learning_rate`: Learning rate for optimization

For full configuration options, see `experiments/configs/default_config.py`.

## Results

The system tracks and visualizes several metrics:

1. **Domain Accuracy**: Accuracy for each domain as training progresses
2. **Forgetting**: Measure of knowledge loss for previous domains
3. **Backward Transfer**: Measure of improvement in previous domains
4. **Average Accuracy**: Overall performance across all domains

## Extending the Project

- Add new continual learning techniques (e.g., Generative Replay, Progressive Neural Networks)
- Experiment with different base models (e.g., RoBERTa, DistilBERT)
- Explore additional datasets and domains
- Implement dynamic memory management techniques

## Citation

If you use this code for your research, please cite:

```
@misc{continual-learning-text-commands,
  author = {Your Name},
  title = {A Continual Learning Approach for Sequential Text Command Understanding},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/continual-learning-project}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.#   M L  
 