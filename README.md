# Hacking the Data Science Job Description

An NLP exploration of the data science labor market. This project scrapes,
cleans, and models ~40,000 Indeed postings to surface (a) which language
distinguishes different data science roles and (b) what latent structure
exists in the posting corpus.

## The question

Data-science job titles are famously inconsistent. A "Data Scientist" at
company A might do what a "Machine Learning Engineer" does at company B and
what an "Analytics Engineer" does at company C. Can we build a model that
looks at a job *description* and predicts the set of role descriptors in its
title — and can we use that model to surface the vocabulary that actually
distinguishes roles from each other?

## Approach

| Stage             | Tools                                                          |
|-------------------|----------------------------------------------------------------|
| Ingest            | 7 CSV exports from scraped Indeed postings                     |
| Clean             | NLTK lemmatization + layered stopword lists + encoding cleanup |
| Target            | Multi-label: top-150 title tokens per posting                  |
| Features          | TF-IDF over descriptions (1,000 max features)                  |
| Classification    | `OneVsRestClassifier` sweep across 12 classifiers              |
| Tuning            | `GridSearchCV` on `LinearSVC` with 5-fold CV                   |
| Topic modeling    | NMF, 18 components                                              |
| Clustering        | MiniBatch K-Means with elbow-plot-selected k                   |
| Visualization     | PCA (2D) and PCA→t-SNE (2D) colored by cluster                 |

## Metrics

- **Row-wise Jaccard similarity** — rewards getting the set right
- **Hamming loss** — penalizes every incorrect bit individually

## Files

```
Project_5_Cline.ipynb      Main notebook
data/                      Input CSVs (git-ignored)
  └── Final Project 5 NLP Data_1(1).csv … Data_3.csv
  └── word_removal.txt     Optional domain-specific words to suppress
artifacts/                 Cached intermediate results (created on first run)
```

## Running it

1. Clone the repo.
2. Place the seven scraped CSVs under `data/`.
3. Create a virtualenv and install dependencies:
   ```
   pip install pandas numpy scikit-learn nltk wordninja tqdm matplotlib seaborn
   ```
4. Open `Project_5_Cline.ipynb` and run top-to-bottom.

## Findings (summary)

- The top four classifiers by Jaccard were `LinearSVC`, `LogisticRegression`,
  `PassiveAggressiveClassifier`, and the 1000-tree Random Forest.
- After `C` tuning, `LinearSVC` reaches moderate Jaccard with low Hamming
  loss on the held-out set.
- Per-class feature introspection (top 10 coefficients per label) shows the
  language that distinguishes each role family — `model`, `training`,
  `pipeline` for machine-learning postings; `system`, `architecture`,
  `deploy` for engineering postings; `dashboard`, `stakeholder`, `report`
  for analyst postings.
- NMF surfaces 18 topic clusters that partly overlap with the K-Means
  assignments — the overlap is the real signal; the divergence is model-
  specific noise.

## Caveats

- **Selection bias.** We only observe postings that made it onto Indeed. The
  vocabulary of postings that *should* exist but don't is unknowable here.
- **Time-boxed.** The scrape is a snapshot. Job market language shifts.
- **English only.** Postings in other languages were filtered out before
  modeling.

## What I'd do differently today

- Replace bag-of-words with sentence-transformer embeddings (all-MiniLM-L6-v2
  or similar). Would likely move Jaccard ceiling up a few points and would
  also produce a more natural similarity-based clustering.
- Switch from K-Means to HDBSCAN. K-Means forces me to pick `k`, assumes
  spherical clusters, and handles density variation poorly — all three are
  wrong for text data.
- Enrich with salary data from a second source (Levels.fyi, Glassdoor) and
  frame regression and classification as joint problems.

---

_Original project: Metis Data Science Bootcamp Capstone, March 2021._
_Cleaned for portfolio presentation, 2026._
