# TODOs

- [ ] Create a better evaluation system, there are several models with different configs, maybe we can use a strategy pattern to implement all of this
- [ ] Fix bugs with DORIS dataset
- [ ] Add multi-task learning per dataset (predict relevance + rating + engagement/watch_percentage for MARS, app_usability/data_quality for ITM, etc.) using shared GNN+Transformer body with task-specific heads and uncertainty-based loss weighting
- [ ] Add fairness (3 approaches, generic algorithms + per-dataset config):
  - [ ] **Pre-processing** (User Demographic Fairness — sample re-weighting):
    - MARS: `user_language` (from EN/FR split)
    - ITM: `age` (age groups)
    - DORIS: `education` or `major` (education level)
  - [ ] **In-processing** (Item Exposure Fairness — popularity bias regularization in loss):
    - MARS: explicit `nb_views` column
    - ITM/DORIS: implicit popularity via interaction count per item (computed from train set)
  - [ ] **Post-processing** (Item Diversity Fairness — constrained difficulty re-ranking):
    - MARS: explicit `difficulty` column
    - DORIS: `grade` as proxy (discretize to easy/medium/hard)
    - ITM: infer difficulty from avg rating per item or `duration` if available
- [ ] Add XAI
