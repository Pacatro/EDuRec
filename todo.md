# TODOs

List of TODOs for the project.

## Code

- [x] Separete prerpocessing logic from data management logic
- [x] Reorganize code
- [ ] Include new models
- [ ] Improve evaluations
- [ ] Improve stadistics tests

## Datasets

Posible datasets to use for the project:

- [x] [MARS](https://www.sciencedirect.com/science/article/pii/S2352340923000604)
- [x] [ITM](https://www.kaggle.com/datasets/irecsys/itmrec?select=items.csv)
- [ ] [Open MOOC Review](https://www.sciencedirect.com/science/article/pii/S0957417422015081) --> Requests to the authors
- [ ] [OULAD](https://www.kaggle.com/datasets/rocki37/open-university-learning-analytics-dataset?select=courses.csv) --> No texts

## Deep Learning models to implement for comparison

See [`recbole`](https://recbole.io/index.html) for implementations.

## Ideas for improvements to the actual model

Some ideas for improvements to the model:

### Change approach

Instead of only predicts the ratings, use Multi-Task Learning to predict other interesting features (relevant and rating).

### Change the way to compute the threshold

El umbral utilizado para determinar si un item es relevante o no se calculará usando la media de los ratings de ese usuario, en lugar de la media de todo el dataset.

- [Transformed based recommender system](https://www.nature.com/articles/s41598-025-08931-1)
