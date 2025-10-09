# EDuRec

This repository is part of the Master’s Thesis (TFM) by Francisco de Paula Algar Muñoz at the Menéndez Pelayo International University.

The goal of this project is to develop a recommendation system for e-learning based on a benchmark dataset, allowing evaluation of its performance compared to previous models.

## Features

* **Hybrid Model**: Implements a neural hybrid model combining collaborative filtering and content-based features.
* **State-of-the-Art Comparison**: Evaluates and compares the model against well-known algorithms from the `surprise` library.
* **Command-Line Interface**: Provides a simple and powerful CLI for training, evaluating, and getting predictions.
* **Multiple Datasets**: Supports two different e-learning datasets: MARS and ITM.
* **Statistical Analysis**: Includes tools for performing statistical tests (Friedman and Nemenyi) to compare model performance.

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

The application is structured into three main commands: `train`, `eval`, and `predict`.

### Train

Train the `NeuralHybrid` recommendation model.

```bash
uv run edurec train [OPTIONS]
```

**Example:**

Train the model on the `mars` dataset for 20 epochs with a batch size of 64, and save it to `mymodel.pt`.

```bash
uv run edurec train --dataset mars --epochs 20 --batch-size 64
```

### Evaluate

The following commands are available for evaluation:

* `eval`: Evaluates the proposed `NeuralHybrid` model using cross-validation.
* `stats-test`: Performs statistical tests to compare model performances.

**`eval` Example:**

Run a 5-fold cross-validation on the `itm` dataset.

```bash
uv run edurec eval --dataset itm --n_splits 5
```

**`stats-test` Example:**

Run Friedman and Nemenyi statistical tests for top-10 results.

```bash
uv run edurec stats --top_k 10
```

### Predict

Make predictions (recommendations) using a trained model.

```bash
uv run edurec predict [OPTIONS]
```

**Example:**

Get the top 5 recommendations using the model `mymodel.pt` on the `mars` dataset.

```bash
uv run edurec predict --model-path mymodel.pt --dataset mars --top-k 5
```

## Project Structure

```
/
├── data/                 # Datasets
├── edurec/               # Main source code
│   ├── commands/         # CLI commands (train, eval, predict)
│   └── core/             # Core logic (model, engine, datamodule, etc.)
├── notebooks/            # Jupyter notebooks for analysis
├── results/              # Folder for evaluation metrics and plots
├── scripts/              # Helper scripts
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Author

[**Francisco de Paula Algar Muñoz**](https://github.com/Pacatro)

<!-- ## Advisors -->
<!---->
<!-- **Amelia Zafra Gómez** -->
<!-- **Cristóbal Romero Morales** -->

## License

[**MIT**](https://opensource.org/license/mit) - Created by [**Paco Algar Muñoz**](https://github.com/Pacatro)
