# EDuRec

EDuRec is a PyTorch Lightning recommendation system for e-learning datasets. It
combines a schema-derived knowledge graph, item text representations, and
sequential user history to recommend educational resources.

This repository is part of the Master's Thesis by Francisco de Paula Algar
Munoz at the Menendez Pelayo International University.

## Project Description

The project implements and evaluates a hybrid educational recommender for
implicit and explicit student-resource interactions. The codebase includes:

- Dataset loaders and preprocessing for `mars`, `itm`, and `doris`.
- The proposed EDuRec model, implemented with PyTorch, PyTorch Geometric, and
  Lightning.
- Training, testing, dataset inspection, hyperparameter optimization, benchmark
  evaluation, and ablation commands through a Typer CLI.
- RecBole-based comparisons against classical and state-of-the-art recommenders.
- Ranking metrics at multiple cutoffs, including Precision, Recall, NDCG, Hit
  Rate, MAP, and MRR.

## Installation

The project uses Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/) for
dependency management.

```bash
git clone https://github.com/Pacatro/EDuRec.git
cd EDuRec
uv sync
```

For development dependencies such as `pytest` and `mlflow`, use:

```bash
uv sync --group dev
```

The repository expects raw datasets under `data/raw/<dataset>`. The included
loaders currently support:

- `data/raw/mars`
- `data/raw/itm`
- `data/raw/doris`
- `data/raw/mooccubex`
- `data/raw/coco`
- `data/raw/synthetic` (generated locally, see below)

### Synthetic Dataset

A deterministic synthetic dataset with rich free-text features on users,
items and interactions can be generated on the fly:

```bash
uv run edurec synthetic
```

It writes `users.csv`, `items.csv` and `interactions.csv` under
`data/raw/synthetic` (about 8k users, 1.2k items and 300k explicit
interactions). The first time a command needs it, the `synthetic` loader
generates the files automatically. Use `--force` to regenerate and
`--num-users`, `--num-items`, `--target-interactions` and `--seed` to change
the size or the random seed:

```bash
uv run edurec synthetic --num-users 2000 --num-items 400 \
  --target-interactions 60000 --seed 7 --force
```

## Usage

All commands are exposed through the `edurec` CLI.

```bash
uv run edurec --help
```

Global options:

```text
-d, --device [auto|cpu|cuda]  Device to use
-r, --random-state INTEGER    Random seed
-v, --verbose                 Verbose output
-h, --help                    Show help
```

### Inspect a Dataset

Print basic statistics and sample rows from a dataset.

```bash
uv run edurec dataset --dataset explicit_mars --max_rows 10
```

Options:

```text
-d, --dataset [explicit_mars|implicit_mars|itm|doris|mooccubex|coco|synthetic]  Dataset to use
-m, --max_rows INTEGER          Number of rows to show
```

### Train EDuRec

Train the proposed model on one dataset. If `--dataset` is omitted, the command
iterates through all registered datasets.

```bash
uv run edurec train --dataset explicit_mars --use_processed --save_model
```

Common options:

```text
-d, --dataset [explicit_mars|implicit_mars|itm|doris|mooccubex|coco|synthetic]
-e, --epochs INTEGER            Default: from saved train config
-l, --lr FLOAT                  Default: from saved train config
-b, --batch_size INTEGER        Default: from saved train config
-p, --patience INTEGER          Default: from saved train config
-v, --val_size FLOAT            Default: 0.1
-t, --test_size FLOAT           Default: 0.2
-k, --top_k INTEGER             Default: 20
-R, --remove_sparse             Remove sparse users/items
-i, --min_interactions INTEGER  Default: 3
-a, --adaptive_k                Use adaptive-k metrics where supported
-D, --debug                     Fast debug run
-S, --save_model                Save checkpoint, config, and metrics
-P, --use_processed             Reuse cached processed data
-M, --models-folder TEXT        Default: models
-C, --configs-folder TEXT       Default: configs
-E, --experiment-name TEXT      Optional logger experiment name
```

### Test a Saved Model

Load the most recent saved model for a dataset and evaluate it on the test
split.

```bash
uv run edurec test --dataset explicit_mars --use_processed
```

Options include dataset, batch size, validation/test split sizes, top-k,
adaptive-k, sparse filtering, and the models folder.

### Evaluate EDuRec and SOTA Models

Run the proposed EDuRec model and RecBole baselines on the selected dataset. If
`--dataset` is omitted, all datasets are evaluated.

```bash
uv run edurec eval --dataset itm --use-processed --top-k 5 --top-k 10 --top-k 20
```

Default SOTA models:

- `ItemKNN`
- `NeuMF`
- `LightGCN`
- `MultiVAE`
- `SGL`
- `SASRec`
- `BERT4Rec`

Useful options:

```text
-d, --dataset [explicit_mars|implicit_mars|itm|doris|mooccubex|coco|synthetic]
-e, --epochs INTEGER            Default: from saved train config
-l, --lr FLOAT                  Default: from saved train config
-b, --batch-size INTEGER        Default: from saved train config
-p, --patience INTEGER          Default: from saved train config
-k, --top-k INTEGER             Repeat for multiple cutoffs
-R, --remove-sparse / -K, --keep-sparse
-I, --min-interactions INTEGER  Default: 3
-P, --use-processed / -N, --no-use-processed
-c, --cfg-path FILE             Extra RecBole config
-m, --sota-model TEXT           Repeat to choose baseline models
-a, --adaptive-k / -A, --fixed-k
```

Results are written to `results/evaluations/<dataset>/`, with one CSV per model
and seed, the detailed `evaluation_results.csv`, and the aggregated
`evaluation_summary.csv`.

### Optimize Hyperparameters

Run Optuna-based hyperparameter optimization for EDuRec.

```bash
uv run edurec optim --dataset explicit_mars --trials 30 --use_processed
```

The command saves the best model and training configurations, trial log, and
study database under `results/optimization/<dataset>/`. It also writes the best
configuration for each dataset and architecture to
`configs/model/<dataset>_<arch>.yaml` and `configs/train/<dataset>_<arch>.yaml`,
so they can be reused by training and evaluation. Use `--configs-folder` to
choose a different folder.

### Saved Configurations

The `configs/` folder keeps one model configuration and one independent training
configuration per evaluated dataset **and architecture**:

```text
configs/model/<dataset>_<arch>.yaml   Model architecture hyperparameters
configs/train/<dataset>_<arch>.yaml   Training hyperparameters (epochs, lr,
                                      batch size, patience, weight decay, top-k,
                                      adaptive-k)
```

`<arch>` can be `kg_rnn`, `kg_transformer`, or `sasrec_text`. When a config file exists for the
dataset and architecture being run, training, evaluation, and ablation commands
load it. Explicit CLI flags always take precedence over the saved
configurations, which in turn take precedence over the global defaults in
`edurec/settings.py`.

The model config's `scorer_type` can be `mlp`, `dot`, or `candidate_attention`.
The candidate-attention scorer attends from each candidate course to the
encoded history and ranks it using the resulting candidate-specific context.

### Run Ablations

Evaluate EDuRec variants across multiple random seeds.

```bash
uv run edurec ablation --dataset doris --seeds 13,42,77,101,2026 --use_processed
```

Implemented main variants:

- `full`: full EDuRec architecture.
- `no_graph`: drops the knowledge-graph structure (attribute nodes and
  message passing), keeping only item ID embeddings and feature projections.
- `no_text`: removes the text embeddings from the item node features.
- `no_item_bias`: removes the learned item-popularity bias.
- `dot_product`: replaces the MLP scorer with dot-product scoring.
- `candidate_attention`: attends from each candidate course to the encoded
  history before scoring the user-course pair.

Variants that disable a module the dataset does not provide (for example
`no_text` on a dataset without text features) are marked as not applicable and
excluded from the plots. Aggregated outputs are saved to
`results/ablations/<dataset>/`.

## Model Architecture

![EDuRec model architecture](model-diagram.png)

EDuRec exposes `kg_rnn` and `kg_transformer` through the `arch` field of the
model configuration. Both architectures refine item embeddings with the
knowledge graph and encode each user's chronological history with either a
GRU/LSTM or Transformer. The `scorer_type` setting selects an MLP, dot product,
or candidate-conditioned attention scorer.

The sections below describe the modules.

The `sasrec_text` architecture implements a content-enhanced SASRec:
`course_emb = ID embedding + projected content features`. Content features reuse
the preprocessing pipeline's frozen pretrained text embeddings (text fields
are joined with their field names) and numeric course metadata. Both history
and candidates use the same course embedding table. A causal Transformer with
learned positions produces the user state from the last valid history position;
dot products with candidates produce scores for Top-K ranking. Graph, user
profile, attention pooling, and item bias are disabled for this architecture.
Empty histories produce a zero state and tied scores. If no content features
are available, the model uses course ID embeddings alone. Only `full` and
`no_text` (when text is available) apply in the ablation command.

```bash
uv run edurec train --dataset doris --arch sasrec_text
```

Use the existing `transformer_hidden_dim`, `transformer_layers`,
`transformer_heads`, `max_history_len`, `emb_dim`, and `dropout` model settings.

- **Knowledge-graph encoder**: a heterogeneous item-item graph is derived from
  each dataset schema. Items are nodes, and every categorical or list-valued
  item field becomes an attribute node type. Each item connects to its
  attribute values through an edge named after the field, so items that share
  an attribute value become neighbours through that shared attribute node.
  User-item interactions are not modeled, and the graph contains no user nodes.
  Extra relations are declared per dataset in the schema and resolved by the
  same generic builder: `refs` links fields whose values name another entity
  (for example DORIS course prerequisites) and `cooc` links two attributes that
  co-occur in a row (for example COCO category levels). Reverse edges and edge
  cleanup are delegated to PyTorch Geometric. Numeric and text embeddings
  initialize the item nodes.
- **Sequential encoder**: a GRU encodes each user's recent item history. Because
  the graph only contains items, this sequence is the sole source of user
  representations, so a chronological timestamp is required.
- **Scorer**: `scorer_type` selects an MLP, dot product, or candidate-attention
  scorer. Candidate attention queries the encoded history with each course
  candidate and scores the user, candidate, and attended context together. An
  optional item bias can be added.

Module availability is inferred from each processed dataset when the model
configuration is built. The knowledge graph automatically reflects the fields
declared by each dataset schema. Sequential history requires a real
chronological interaction field; datasets without one cannot train the model.

## Implemented Experiments

### Proposed Model Evaluation

`uv run edurec eval` trains EDuRec, evaluates the best checkpoint on the test
split, and reports Precision, Recall, NDCG, Hit Rate, MAP, and MRR at the
configured top-k values.

Explicit feedback trains a rating-weighted ranking loss over the full catalog.
All ratings are retained: training min/max ratings map to weights in [0.1, 1],
then weights are divided by their training mean. Higher ratings contribute more;
low ratings remain weak positive observations, not explicit negative targets.
Validation/test loss uses the same training-fitted scale with clipping. Implicit
events have unit weight and retain sampled-negative training. No binary
`relevant` column or user-mean rating filter is used. Histories and graph edges
include every retained interaction, including low-rated courses.

Retrieval metrics remain unweighted: they measure recovery of each held-out
observed interaction, rather than graded rating preference. RecBole baselines
receive all interactions but retain their own training objectives. Existing
preprocessing caches must be regenerated (cache format version 5); results from
the former positive-only evaluation protocol are not directly comparable.

### SOTA Benchmark Evaluation

The same evaluation command exports RecBole atomic files and runs the baseline
models with aligned split files, learning rate, epoch count, patience, batch
size, and top-k settings. Sequential baselines receive prebuilt histories for
the original train, validation, and test splits instead of asking RecBole to
split the merged interaction file again.

Final ranking metrics use one shared evaluator for every model. Each observed
test interaction is one query with one target, and EDuRec and the RecBole
baselines use the same full item catalog, seen-item mask, and TorchMetrics
implementations for Precision, Recall, NDCG, Hit Rate, MAP, and MRR.

### Hyperparameter Optimization

`uv run edurec optim` runs Optuna studies for EDuRec and saves the best model
and training configurations as YAML for later training, evaluation, or ablation
experiments.

### Ablation Study

`uv run edurec ablation` evaluates architecture variants across configurable
seeds and records metrics, parameter counts, and per-run configuration files.
This is intended to isolate the contribution of the knowledge-graph structure,
text features, item bias, and the scoring function.

## Author

[Francisco de Paula Algar Munoz](https://github.com/Pacatro)

## Advisors

Amelia Zafra Gomez

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
