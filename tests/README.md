# Basic Tests

Run from the project root, with the development dependencies installed:

```sh
uv run python -m pytest -q
```

The suite uses small synthetic datasets on CPU, without downloading datasets or weights.
It covers temporal splits, histories, and context without information leakage;
processed data save/load, atomic replacement and legacy caches;
configuration and ablations; pooling, encoders, scoring, and the contrastive loss;
forward/backward passes of KGSeq and SASRecText; ranking metrics; and CLI input.
Preprocessing tests also cover integer ID alignment, train-only statistics,
empty numeric/categorical fields, normalized lists, and shared text encoding
using a fake embedding model. An integration test follows raw tables through
cache reload, graph and Torch views, model forward/backward, and RecBole export.

It checks the essential contracts, without running full training,
hyperparameter optimization, or benchmarks against external services.
