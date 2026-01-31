# EDuRec

This repository is part of the Master's Thesis (TFM) by Francisco de Paula Algar Muñoz at the Menéndez Pelayo International University.

The goal of this project is to develop a Deep Learning-based educational recommendation system that optimizes the learning experience through intelligent resource selection. The system uses deep neural networks to capture complex patterns in student interaction data, considering both academic history and preferences to generate accurate and adaptive recommendations. Performance is validated using standard metrics (Precision@K, Recall@K, NDCG) and compared against classical and state-of-the-art (SOTA) approaches.

## Features

- **Deep Learning Model**: Implements EDuRec, a neural recommendation model using PyTorch Lightning with configurable architectures.
- **Cross-Validation**: Supports multiple cross-validation strategies (K-Fold, Stratified K-Fold) for robust evaluation.
- **State-of-the-Art Comparison**: Evaluates and compares the model against algorithms like DeepFM using the RecBole framework.
- **Statistical Analysis**: Includes Friedman and Nemenyi tests to compare model performances statistically.
- **Command-Line Interface**: Provides a powerful CLI built with Typer for training, evaluation, and analysis.
- **Multiple Datasets**: Supports two e-learning datasets: MARS and ITM.
- **Experiment Tracking**: Integrates Weights & Biases for experiment logging and monitoring.

## Getting Started

> [!NOTE]
> To run this project, you need to have the [`uv`](https://docs.astral.sh/uv/) package manager installed.

Follow these steps to run the project:

1. **Clone the repository**

   ```bash
   git clone https://github.com/Pacatro/edurec.git
   cd edurec
   ```

2. **Install dependencies and create a virtual environment**

   ```bash
   uv sync
   ```

3. **Run the application**

   To see all available commands and options, run:

   ```bash
   uv run edurec --help
   ```

## Usage

The application provides a main CLI with subcommands for training and evaluation.

### Global Options

- `--device, -d`: Device to use (auto, cpu, cuda)
- `--random-state, -r`: Random state for reproducibility (default: 42)
- `--verbose, -v`: Enable verbose mode

### Train

Train the recommendation model.

```bash
uv run edurec train [OPTIONS]
```

**Options:**

| Option                | Description                     | Default  |
| --------------------- | ------------------------------- | -------- |
| `--dataset, -d`       | Dataset to use (mars, itm)      | mars     |
| `--epochs, -e`        | Number of training epochs       | 100      |
| `--lr, -l`            | Learning rate                   | 0.001    |
| `--batch_size, -b`    | Batch size                      | 256      |
| `--val_size, -v`      | Validation set ratio            | 0.1      |
| `--test_size, -t`     | Test set ratio                  | 0.2      |
| `--top_k, -k`         | Top-k value for recommendations | 10       |
| `--monitor, -m`       | Metric to monitor               | val_loss |
| `--use_logger, -L`    | Use W&B logger                  | False    |
| `--debug, -D`         | Debug mode (fast dev run)       | False    |
| `--save_model, -S`    | Save the trained model          | False    |
| `--save_data, -P`     | Save processed data             | False    |
| `--models-folder, -M` | Folder to save models           | models   |

**Example:**

Train the EDuRec model on the `mars` dataset for 50 epochs with a batch size of 128:

```bash
uv run edurec train --dataset mars --epochs 50 --batch_size 128 --save_model
```

### Evaluate

Evaluate model performance using cross-validation and statistical tests.

```bash
uv run edurec eval [OPTIONS]
```

#### eval subcommand

Evaluates the EDuRec model (and optionally SOTA models) using cross-validation.

```bash
uv run edurec eval eval [OPTIONS]
```

**Options:**

| Option             | Description                               | Default  |
| ------------------ | ----------------------------------------- | -------- |
| `--eval-sota, -S`  | Also evaluate SOTA models (DeepFM)        | False    |
| `--dataset, -d`    | Dataset to use                            | mars     |
| `--batch-size, -b` | Batch size for training                   | 256      |
| `--top-k, -k`      | Top-k value for recommendations           | 10       |
| `--epochs, -e`     | Number of training epochs                 | 100      |
| `--n-splits, -n`   | Number of CV splits                       | 5        |
| `--patience, -p`   | Patience for early stopping               | 10       |
| `--delta`          | Minimum improvement for early stopping    | 0.001    |
| `--monitor, -m`    | Metric to monitor                         | val_loss |
| `--cv-type`        | Cross-validation type (kfold, stratified) | kfold    |
| `--results-folder` | Folder to save results                    | results  |

**Example:**

Run a 5-fold cross-validation on the `itm` dataset:

```bash
uv run edurec eval eval --dataset itm --n_splits 5
```

Evaluate with SOTA comparison:

```bash
uv run edurec eval eval --dataset mars --eval-sota --epochs 50
```

#### stats subcommand

Performs Friedman and Nemenyi statistical tests to compare model performances.

```bash
uv run edurec eval stats [OPTIONS]
```

**Options:**

| Option        | Description | Default |
| ------------- | ----------- | ------- |
| `--top_k, -k` | Top-k value | 10      |

**Example:**

Run statistical tests for top-10 recommendations:

```bash
uv run edurec eval stats --top_k 10
```

## Project Structure

```
/
├── edurec/                  # Main source code
│   ├── cli/                 # CLI commands (train, eval)
│   │   ├── train.py         # Training command
│   │   └── eval.py          # Evaluation and stats commands
│   ├── training/            # Training logic
│   │   ├── engine.py        # RecSys training engine
│   │   ├── model.py         # EDuRec model definition
│   │   └── io.py            # Model I/O operations
│   ├── datasets/            # Data handling
│   │   ├── datamodule.py    # PyTorch Lightning DataModule
│   │   ├── loaders.py       # Dataset loaders
│   │   ├── data_processor.py # Data preprocessing
│   │   └── utils.py         # Dataset utilities
│   ├── evaluation/          # Evaluation metrics and methods
│   │   ├── cross_validation.py # Cross-validation logic
│   │   ├── cv_datamodule.py # CV DataModule
│   │   └── stats.py         # Statistical tests (Friedman, Nemenyi)
│   ├── config.py            # Configuration settings
│   └── main.py              # CLI entry point
├── data/                    # Datasets
├── tests/                   # Unit tests
├── results/                 # Evaluation results
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Author

[**Francisco de Paula Algar Muñoz**](https://github.com/Pacatro)

## Advisors

**Amelia Zafra Gómez**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
