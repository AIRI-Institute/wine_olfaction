# WineSmell: an e-nose wine olfaction collection for retrieval and classification

This repository is the resource accompanying the ECIR 2027 Resource Track submission
*Developing Smell-Aware Intelligence with a Large Wine Olfaction Dataset*
([call for resource papers](https://www.ecir2027.co.uk/call-for-resource-papers)).
It contains the data, loaders, and code that reproduce the tables in the
"End-to-end and retrieval comparison" section of the paper.

WineSmell pairs multivariate electronic-nose (e-nose) traces with expert sommelier
annotations for 551 wines: 13 chemo-resistive channels, 189 time points per measurement,
numeric quality scores from a multi-expert panel, and free-text tasting notes.
It supports two families of methods on the same tasks:

- **end-to-end classification** (classical and neural models trained on the traces), and
- **content-based retrieval** (a query trace is labeled from its nearest gallery traces:
  kNN on PCA features, cosine similarity of self-supervised and curve-fit embeddings).

Because the sensor array drifts across the six-day campaign and within each day,
the repository ships a **leave-one-day-out (LODO)** protocol in which query and gallery
come from different days, next to the usual split by wine label.

Public repository: <https://github.com/alexnkorovin/wine_dataset> (permanent archive with DOI: TODO).

## Contents

| Path | Content |
|------|---------|
| `data/raw/sensor_3d_array.npy` | e-nose traces, `float64`, shape (566 measurements, 13 channels, 189 time points) |
| `data/raw/merged_en.csv` | English metadata and annotations, one row per measurement, row `i` matches trace `i` |
| `data/translation/` | Russian-to-English translation tables used to build `merged_en.csv` |
| `configs/` | shared protocol settings (`base.yaml`) and one file per task |
| `src/` | pipeline: loading, splits, models, metrics, tables |
| `scripts/` | entry points: full grid, table rebuild, translation |
| `slurm/` | cluster submission script |
| `LICENSE` | MIT |

`data/raw/merged.csv` is the Russian working copy with taster names in the column
headers, and `data/private/` holds the taster key. Neither is part of the public release.

## Dataset card

### Collection

- **Event.** The data were collected during an official wine quality evaluation event run
  under the standards of the International Organisation of Vine and Wine (OIV). The e-nose
  measured the same samples that the sommelier panel judged.
- **Campaign.** There are six measurement days, 2025-09-08 to 2025-09-13, with 81 to 105
  measurements per day. Each day is organised into shifts and sets (columns `shift` and
  `set`), and `startTime` and `setTime` give the timing of every measurement.
- **Sensor array.** A multi-electrode lab-on-chip with 13 metal-oxide channels:
  - 3 ZnO
  - 3 ZnO/In
  - 3 ZnO/Ni
  - 2 ZnO/Ce
  - 2 ZnO/Ag
  
  The sensors are heated to 200-400 C under closed-loop temperature control.
- **Protocol.** Each sample sits under a sealed glass dome. Headspace is drawn to the array
  at 3.3 mL/s, and resistance is sampled at 0.4 Hz. Each recording runs as follows:
  - a 10 s baseline;
  - 180 s of exposure, from t = 10 s to t = 190 s;
  - recovery until t = 490 s.
  
  This gives 189 time points per channel. Clean air restores the baseline between samples.
- **Wines and repeats.** There are 566 measurements of 551 distinct wines. Fifteen wines
  were measured twice.

### Annotations

- **Quality score.** A panel of professionals (wine producers, winery CEOs, PhD holders
  in biological and technical sciences, critics and competition winners) scored the wines on
  the 100-point Italian Sommelier Association (ISA) scale. The scale covers appearance, aroma,
  taste and harmony. `wine_quality` is the mean score, with mean 82.3 and range 61.3 to 90.9.
  Individual scores are kept in the 29 anonymised columns `R01`..`R29`. Each wine has 6 to 13
  scores (median 11), and 2 wines have none. Per-wine panel
  statistics are in `n_reviewer_scores`, `reviewer_mean`, `reviewer_median`, `reviewer_std`,
  `reviewer_min`, `reviewer_max` and `reviewer_range`.
- **Tasting notes.** For each wine, a single designated sommelier wrote a note using the ISA
  vocabulary. Three text fields hold the notes:
  - `digest_description`: colour, nose, palate and food pairing (515 rows);
  - `good_description`: a one-line verdict of the wine's merits (499 rows);
  - `tech_description`: the producer's technical sheet (wine type, alcohol, vintage, grapes, sugar).
- **Aroma labels.** Three label sets are derived from the notes:
  - `aroma_labels`, `aroma_labels_chatgpt`: aroma families extracted by an LLM;
  - one-hot descriptors `cherry` ... `grass` with `aroma_targets` and `aroma_target_ids`;
  - six coarse markup clusters `aroma_markup_*` (berries, tree fruit, citrus, dried fruit,
    fruit not mentioned, bad), with the first set cluster given in `aroma_markup_label`.

### Columns of `merged_en.csv`

| Group | Columns |
|-------|---------|
| Identity | `no`, `code` (wine index 1..551), `ID_code` (catalog code, e.g. `351RSK0071`) |
| Wine metadata | `name` (style), `year` (vintage), `raw` (grapes), `sugar` (g/dm3), `color`, `sweetness`, `strength_low`, `strength_high`, `strength_mean` (alcohol as a fraction), `strength_bins` (0-4) |
| Measurement | `date`, `shift`, `set`, `number_of_samples`, `startTime`, `setTime` |
| Quality | `wine_quality`, `wine_quality_low_mean`, `wine_quality_low_mean_minus_{1..4}std` (1 if the score is below the mean minus k std) |
| Panel | `R01`..`R29`, `n_reviewer_scores`, `reviewer_*` |
| Text | `tech_description`, `digest_description`, `good_description`, `has_digest` |
| Aroma | `aroma_labels*`, `aroma_labels_bad_wine_flags`, descriptor one-hots, `aroma_targets`, `aroma_target_ids`, `aroma_markup_*` |

`code` and `ID_code` describe the same wine: there are 551 values of each, the match is
one-to-one, and `code` numbers the wines in `ID_code` order. Splits and deduplication use `code`.

### Preparation

1. Trace files were stacked into `sensor_3d_array.npy` in measurement order.
2. Wine metadata, panel scores and notes were joined to measurements by catalog code.
   Internal join and debug fields were dropped.
3. `color` and `sweetness` were derived from the style keywords in `name`.
   `strength_*` values were parsed from the strength field, and `strength_bins` are
   equal-frequency bins.
4. `scripts/translate_merged.py` builds `merged_en.csv` from the Russian working copy:
   - free text is translated through the batch files in `data/translation/`;
   - styles, grape names and technical sheets go through a term glossary (`glossary.json`);
   - taster columns are renamed `R01`..`R29`;
   - `ID_code` is transliterated (`RSK`).
   
   Row order and every non-text column are unchanged. Each task yields the same labels from
   either file.

### Ethics and anonymisation

Sommelier identities are replaced by stable codes `R01`..`R29`, and the code-to-name key is
not distributed. The panel took part in an official evaluation event. Consent for releasing
the anonymised scores: TODO (state the agreement or approval here).

### Known limitations

- **Sensor drift.** Baselines shift across the campaign and within each day (see the
  paper's sensor stability analysis). Traces from the same day share a sensor state, so
  splits by wine label overestimate how well results transfer to new sessions. Use LODO to
  control for this.
- **Imbalance.** Rows are 53% white, 36% red and 10% rose. 75% are dry, and most vintages
  are 2023-2024. The wines come from one national competition, with many local grape
  varieties.
- **Single-author notes.** Each tasting note is the work of one expert, whereas quality
  scores average 6 to 13 experts. Panel agreement: median per-wine std 2.6, median pairwise
  Pearson correlation 0.69.
- **Missing values.**
  - `strength_bins` is missing for 182 rows;
  - `digest_description` is empty for 51 rows;
  - a few notes are truncated in the source.
- **Abbreviations.** The source abbreviations `pr`, `TP`, `ZhB`, `ob` and `or` in `name`
  are kept as tags because their meaning is not documented.
- **Translation.** The English text is a translation. The Russian originals are available
  on request.

## Loading the data

```python
import numpy as np
import pandas as pd

X = np.load("data/raw/sensor_3d_array.npy")          # (566, 13, 189)
meta = pd.read_csv("data/raw/merged_en.csv")          # 566 rows, aligned with X
first = ~meta["code"].duplicated(keep="first")        # one measurement per wine
X, meta = X[first.values], meta[first].reset_index(drop=True)   # 551 wines
print(meta[["code", "date", "color", "wine_quality", "digest_description"]].head())
```

## Tasks

| Task | Config | Target | Classes |
|------|--------|--------|---------|
| Color-2 | `color2.yaml` | `color` | red / white |
| Color-3 | `color3.yaml` | `color` | red / white / rose |
| ABV-b | `abv_b.yaml` | `strength_bins` | 5 alcohol bins |
| Aroma | `aroma_markup_ce.yaml` | first set `aroma_markup_*` cluster | 6 clusters |
| Low-q | `low_q.yaml` | `wine_quality_low_mean` | below / above the mean score (82.2) |

## Protocols

- **5x5 split by wine label (`cv` and `holdout`).**
  1. A 4:1:1 shuffle (seed 42) sets the first sixth of the rows aside as the outer test.
  2. StratifiedKFold(5) on the remaining train and validation rows gives 5 folds.
  3. Each fold is trained with 5 seeds (`42 + 1000*fold + repeat`), 25 fits in total.
  4. Each fit is scored twice: on its left-out fold (`cv`) and on the outer test (`holdout`).
     This gives 25 `cv` scores and 25 `holdout` scores per cell.

  Naive and retrieval models use the same 25 runs. For retrieval, the training rows are the
  gallery and the evaluation rows are the queries.
- **LODO.** Train on five measurement days, or use them as the gallery, and test on the
  sixth (6 runs).
- **Preprocessing.** Keep the first measurement per `code`, z-score each (channel, time point)
  over the task rows, and optionally apply LOWESS (frac 0.05, `lowess` / `no_lowess`). The
  result is cached in `data/processed/{task}_{smooth}.npz`.
- **Drift correction** (`src/drift.py`), fit on the training rows of each run only:
  - `none`;
  - `snorm`: z-score each channel within each measurement day; a day absent from the training
    rows uses the pooled training statistics;
  - `tod_residual`: subtract a ridge fit of every feature on the time of day
    (sin and cos of the `startTime` hour);
  - `day_residual`: subtract a ridge fit of every feature on a one-hot of the day.
- **Metrics.** Accuracy, F1 and ROC-AUC are reported as mean +/- population std over runs.
  F1 is binary for 2-class tasks and macro otherwise. ROC-AUC is one-vs-rest and is undefined
  for `svm`.

## Models

- Naive: stratified dummy classifier.
- Classical, on flattened traces: `rf`, `xgb`, `plsda`, `svm`.
- Neural: `MLSTMfcn`, `Transformer`, `CNNLSTM4`, `CNN1DClassifier`, `LSTMNet` and
  `EmissionCNN1DLSTM`. All are trained for 300 epochs with Adam (lr 1e-4), batch size 1000,
  and BCE on one-hot targets.
- Retrieval: `knn_pca` (kNN vote on 32 PCA components), `cosine_ssl` and `cosine_curvefit`
  (k = 1).

## Reproducing the paper tables

```bash
pip install -r requirements.txt
python scripts/run_all.py                 # full grid + tables (days on one CPU)
python scripts/write_tables.py            # rebuild tables from results/runs/
python scripts/run_all.py --tasks color2 --models rf knn_pca --drifts none --epochs 1 --repeats 1  # smoke test
```

The grid is 5 tasks x 14 models x {`lowess`, `no_lowess`} x {`none`, `snorm`,
`tod_residual`, `day_residual`} x {5x5 split, LODO}. Each run writes one file,
`results/runs/parts/{task}__{model}__{smooth}__{drift}/{split}__{run}.csv`, and
`run_all.py` skips runs whose file exists. Several copies can run at once: each run is
claimed with a lock file in `results/runs/claims/`. To run on a Slurm cluster, use
`bash slurm/submit_all.sh`, which starts arrays of GPU and CPU workers on the same queue.

| Step | File | Role |
|------|------|------|
| 1 | `src/step1_load_data.py` | load traces, labels, days and start times; z-score, LOWESS |
| 2 | `src/splits.py` | 5x5 split by wine label with outer test, leave-one-day-out |
| 3 | `src/step3_train.py` | drift correction (`drift.py`), then one fit per run (`classical.py`, `step2_torch.py` + `models.py`, `retrieval.py`) |
| 4 | `src/step4_analyze.py` | accuracy, F1 and ROC-AUC per run; complete cells go to `results/runs/{task}__{model}__{split}__{smooth}__{drift}.csv` |
| 5 | `src/step5_visualize.py` | aggregate runs into `results/tables/*.csv` |

Every table in `results/tables/` exists for each evaluation split `{split}` in `cv`,
`holdout` and `lodo`:

| CSV | Content |
|-----|---------|
| `all_runs.csv` | mean, std and run count for every task, model, split, smoothing and drift |
| `naive_{split}.csv` | Naive baseline in every smoothing x drift cell |
| `classical_{split}.csv` | `rf`, `xgb`, `plsda`, `svm` in every cell |
| `neural_{split}.csv` | the six neural models in every cell |
| `ir_{split}.csv` | the three retrieval models in every cell |
| `best_{split}.csv` | best end-to-end (classical or neural) vs best retrieval model by accuracy, at three grains: `overall` (across all smoothing x drift cells), `cell` (per smoothing x drift) and `model` (best cell of each model) |
| `ir_wins.csv` | metrics on which the best retrieval model beats the best end-to-end model, per split, `overall` and per cell |

All model families here use the single protocol above. The numbers in the submitted PDF came
from earlier scripts, in which classical and retrieval models were scored on fold-test rows and
the LODO neural models used a shorter training schedule. As a result, some cells can differ
slightly from the PDF.

## License, maintenance, citation

- **License.** Code and data are released under the MIT License (`LICENSE`).
- **Maintenance.** Issues and corrections go through the GitHub repository. Releases are
  versioned, and any change to the data comes with a changelog entry and a regenerated
  `merged_en.csv`.
- **Citation:**

```bibtex
@inproceedings{winesmell2027,
  title     = {Developing Smell-Aware Intelligence with a Large Wine Olfaction Dataset},
  author    = {TODO},
  booktitle = {Proceedings of the 49th European Conference on Information Retrieval (ECIR)},
  year      = {2027}
}
```
