# Basic Tests

Run from the project root, with the development dependencies installed:

```sh
uv run python -m pytest -q
```

The suite uses small synthetic datasets on CPU, without downloading datasets or weights.
It covers temporal splits, histories, and context without information leakage;
configuration and ablations; pooling, encoders, scoring, and the contrastive loss;
forward/backward passes of KGSeq and SASRecText; ranking metrics; and CLI input.

It checks the essential contracts, without running full training,
hyperparameter optimization, or benchmarks against external services.
