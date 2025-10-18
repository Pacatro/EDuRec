# TODOs

List of TODOs for the project.

## Datasets

Posible datasets to use for the project:

- [x] [MARS](https://www.sciencedirect.com/science/article/pii/S2352340923000604)
- [x] [ITM](https://www.kaggle.com/datasets/irecsys/itmrec?select=items.csv)
- [ ] [Open MOOC Review](https://www.sciencedirect.com/science/article/pii/S0957417422015081) --> Requests to the authors
- [ ] [OULAD](https://www.kaggle.com/datasets/rocki37/open-university-learning-analytics-dataset?select=courses.csv) --> No texts

## Deep Learning models to implement for comparison

See more interesting models [here](https://www.d2l.ai/chapter_recommender-systems/index.html).

- [x] [MF](https://www.d2l.ai/chapter_recommender-systems/mf.html#model-implementation)
- [ ] [NeuralMF](https://www.d2l.ai/chapter_recommender-systems/neumf.html#the-neumf-model)
- [ ] [Massive Open Online Courses (MOOCs) Recommendation Modeling using Deep Learning](https://www.researchgate.net/publication/338946585_Massive_Open_Online_Courses_MOOCs_Recommendation_Modeling_using_Deep_Learning) --> Classification, very similar to the old proposed
- [ ] [Research on Online Learning Resource Recommendation Method Based on Wide & Deep and Elmo Model](https://www.researchgate.net/publication/338425005_Research_on_Online_Learning_Resource_Recommendation_Method_Based_on_Wide_Deep_and_Elmo_Model)
- [ ] [CBCNN](https://dl.acm.org/doi/10.1007/s00530-017-0539-8)
- [ ] [AutoRec](https://www.d2l.ai/chapter_recommender-systems/autorec.html#model)

## Ideas for improvements to the actual model

- Use Dot product and of Hadamard product
- Use [BPR](https://www.d2l.ai/chapter_recommender-systems/ranking.html#bayesian-personalized-ranking-loss-and-its-implementation) loss
  - To use this, we have to take negative samples of the dataset
- Transformers for attention mechanism in user/item text data

## Actual improvements
