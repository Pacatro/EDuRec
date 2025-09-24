# UcoRecSys

This repository is part of the Bachelor’s Thesis (TFG) by Francisco de Paula Algar Muñoz at the University of Córdoba (UCO), titled: **_Application of Recommendation Systems in Educational Environments_** ([PDF](./Memoria_TFG.pdf)).

> [!NOTE]  
> Most of the project is written in Spanish, only this repository has been translated into English.

The goal of this project is to develop a recommendation system for e-learning based on a benchmark dataset, allowing evaluation of its performance compared to previous models.

## Usage

```bash
ucorecsys [-h] (-i MODEL_PATH | -t | -e | -s | -st) [-ds {mars,itm}] [-cv {kfold,loo}]
                 [--top_k TOP_K] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--output_model MODEL_OUT]
                 [-lr LR] [-k K_SPLITS] [--seeds SEEDS] [-p] [-v]
```

### Options

```bash
-h, --help            Show this help message and exit
-i MODEL_PATH, --inference MODEL_PATH
                      Run inference on a trained model (provide path to model file).
-t, --train           Train the model.
-e, --eval            Evaluate the model.
-s, --surprise        Run Surprise evaluation.
-st, --stats_test     Run statistical tests (default: False).

Common Options:
  -ds {mars,itm}, --dataset {mars,itm}
                        Dataset to use (default: mars).
  -cv {kfold,loo}, --cvtype {kfold,loo}
                        Cross-validation type (default: kfold).
  --top_k TOP_K         Top-k value for ranking metrics (default: 10).

Training Options:
  --epochs EPOCHS       Number of training epochs (default: 50).
  --batch_size BATCH_SIZE
                        Training batch size (default: 128).
  --output_model MODEL_OUT
                        Path to save the trained model (default: model.pt).
  -lr LR                Learning rate (default: 0.001).

Evaluation Options:
  -k K_SPLITS, --k_splits K_SPLITS
                        Number of CV splits (default: 5).
  --seeds SEEDS         Random seeds (default: [0, 1, 42]).

Miscellaneous Options:
  -p, --plot            Generate plots.
  -v, --verbose         Enable verbose output.
```

## Getting Started

> [!NOTE]  
> To run this project, you need to have the [`uv`](https://docs.astral.sh/uv/) package manager installed.

Follow these steps to run the project:

1. **Clone the repository**

   ```bash
   git clone https://github.com/Pacatro/UcoRecSys.git
   cd UcoRecSys
   ```

2. **Install dependencies and create a virtual environment**

   ```bash
   uv sync
   ```

3. **Run the project**

   ```bash
   uv run src/main.py -h
   ```

## Examples

- Train the model on the MARS dataset for 10 epochs, generating a Top-15 recommendation list and saving the model as `tfg.pt`:

  ```bash
  uv run src/main.py -t -ds mars --epochs 10 --top_k 15 --output_model tfg.pt
  ```

- Run inference using the previously trained model:

  ```bash
  uv run src/main.py -i tfg.pt
  ```

## Author

[**Francisco de Paula Algar Muñoz**](https://github.com/Pacatro)

## Advisors

**Amelia Zafra Gómez**  
**Cristóbal Romero Morales**

## License
[**MIT**](https://opensource.org/license/mit) - Created by [**Paco Algar Muñoz**](https://github.com/Pacatro)
