# bt4222-grp8

# Purchase-Oriented Personalised Product Ranking in JDSearch

BT4222 · Group 8 · AY2026/27 Semester 1

**Data:** all data files are in the shared Google Drive folder [Grp8Submission](https://drive.google.com/drive/folders/1dUuMZRnDeOlHbqMCRhEiCvNDM812VPxC?usp=sharing).

For each search in the JDSearch dataset, the project re-ranks the retrieved candidate products by **purchase probability**, so that the product the user actually buys appears near the top of the list.

## Approach

Two models capture different signals, and a gate blends them.

| Component                | What it captures                                                                                | Method                                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Model 1** (short-term) | Relevance: query–product similarity, product popularity, recent item / brand / category history | LightGBM on 17 selected features (from 27 candidates), weighted objective, 3 seeds, Optuna tuning on val1 |
| **Model 2** (long-term)  | Personalisation: the user's long-term behaviour sequences                                       | DIN-style network with target attention over separate click / cart / order channels                       |
| **Gate**                 | How much to trust each model                                                                    | `score = w · P1 + (1 − w) · P2`, with one weight `w` for every user, chosen on val2                       |

Both models output purchase probabilities, calibrated with Platt scaling on val2.

## Results (test set)

Evaluated on the 2,023 test searches that contain a purchase. The purchase is the only relevant item (graded NDCG@10 also credits clicks and carts and is supplementary).

| Model            | NDCG@10 | HitRate@10 | HitRate@1 | MRR    | Mean rank | Median rank | Graded NDCG@10 |
| ---------------- | ------- | ---------- | --------- | ------ | --------- | ----------- | -------------- |
| Model 1          | 0.3972  | 0.6282     | 0.2165    | 0.3447 | 16.68     | 6           | 0.4082         |
| Model 2          | 0.3673  | 0.6139     | 0.1769    | 0.3113 | 17.01     | 7           | 0.3768         |
| Blend (w = 0.60) | 0.4032  | 0.6406     | 0.2155    | 0.3487 | 15.66     | 6           | 0.4140         |

A random order has a HitRate@10 of 0.146 on test. Numbers are from the latest run of `step5_gate_and_evaluation.ipynb`; rerunning with new score files can change them.

## Pipeline

Run the notebooks in this order. Each notebook lists its inputs and outputs in its first cell.

| Step | Notebook                                 | Purpose                                                                                                           |
| ---- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1    | `step1_data_preprocessing.ipynb`         | Cleans the raw data, builds the train / val1 / val2 / test splits and the negative-sampled train set              |
| 2    | `step2_model1_feature_engineering.ipynb` | Builds Model 1 features for every (search, candidate) pair                                                        |
| 2    | `step2_model2_scalar_features.ipynb`     | Builds the scalar (non-sequence) features for Model 2                                                             |
| 2    | `step2_model2_sequence_features.ipynb`   | Builds the user history sequences for Model 2                                                                     |
| 2    | `step2_gate_features.ipynb`              | Builds per-search user features for the learned gate (history length, time since last activity, purchase count)   |
| 3    | `step3_model1_ablation.ipynb`            | Feature screening and ablation for Model 1                                                                        |
| 3    | `step3_model2_ablation.ipynb`            | Ablation of Model 2 components (attention, recency decay, channels, features)                                     |
| 4    | `step4_model1_lightgbm.ipynb`            | Tunes and trains Model 1, calibrates it, saves its scores                                                         |
| 4    | `step4_model2_din.ipynb`                 | Trains Model 2, calibrates it, saves its predictions                                                              |
| 5    | `step5_gate_and_evaluation.ipynb`        | Compares different gate approached on val2, choose the best one. Evaluates Model 1, Model 2 and the blended model |

## Data splits

The split is at row level (about 70% train, 10% val1, 10% val2, 10% test).

| Split | Used for                                                  | Candidate lists                                      |
| ----- | --------------------------------------------------------- | ---------------------------------------------------- |
| train | Fitting the models                                        | Negative-sampled (about 21–22 candidates per search) |
| val1  | Hyperparameter tuning, feature selection, early stopping  | Full (about 89 candidates)                           |
| val2  | Calibration (Platt scaling) and choosing the blend weight | Full                                                 |
| test  | Final evaluation only                                     | Full                                                 |

Only train is negative-sampled, so validation and test metrics are not inflated by short candidate lists. Train metrics are not comparable to val2 and test.

## Evaluation

Every metric is computed per search and averaged over the searches that contain a purchase: NDCG@10 (main), HitRate@10, HitRate@1, MRR, mean and median rank of the first purchase, and graded NDCG@10 (click 1, cart 3, purchase 7).

## Running the notebooks

1. Install the dependencies: `pip install -r requirements.txt`
2. Get the data from the Google Drive folder linked at the top.
3. Point the notebooks at the project data folder:
   - On Google Colab, mount Google Drive (the notebooks do this themselves).
   - Locally, set the environment variable `GRP8_PROJECT` to the folder that contains the step folders (`step1/`, `step2_model1/`, `step4_model1/`, `step4_model2/`, `step5/`, and so on).
4. Run the notebooks in the order above.

## Repository layout

```
.
├── step1_data_preprocessing.ipynb
├── step2_model1_feature_engineering.ipynb
├── step2_model2_scalar_features.ipynb
├── step2_model2_sequence_features.ipynb
├── step2_gate_features.ipynb
├── step3_model1_ablation.ipynb
├── step3_model2_ablation.ipynb
├── step4_model1_lightgbm.ipynb
├── step4_model2_din.ipynb
├── step5_gate_and_evaluation.ipynb
├── requirements.txt
└── README.md
```
