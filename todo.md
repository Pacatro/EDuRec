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

- Mejor cálculo de métricas.

  A la hora de calcular las métricas (Precision@K, Recall@K, etc), necesitamos añadir al dataset interacciones negativas (interacciones de usuarios e ítems que no aparezcan en el dataset).

- Usar BPRLoss en lugar de MSELoss.

  Si añadimos interacciones negativas, el BPRLoss puede llegar a dar mejores resultados.

- Evitar sparsity:

  Cuando no existen muchas interacciones entre los usuarios y los ítems en el dataset, el modelo debe ser capaz de prevenir este problema.

  En la primera versión del modelo, el MLP recibía las intreacciones de los usuarios y las caracteristicas de los ítems, haciendo que aquellos usuarios que no tuvieran interacciones no sean considerados.

  Una posible solución sería que el MLP reciba los embeddings de usuarios e ítem, junto con las interacciones, caracteristicas numéricas y categorícas.

- Usar LayerNorm en lugar de BatchNorm: Para datsets con mucho sparsity, puede ser más ideal

- [Transformed based recommender system](https://www.nature.com/articles/s41598-025-08931-1)
