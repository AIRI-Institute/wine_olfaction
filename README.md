# WineSmell: an e-nose wine olfaction collection for retrieval and classification

This repository is the resource accompanying the ECIR 2027 Resource Track submission
*Developing Sensory Intelligence with a Wine Smell Dataset*
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

Public repository: [https://github.com/AIRI-Institute/wine_olfaction](https://github.com/AIRI-Institute/wine_olfaction) (permanent archive with DOI: TODO).

## Contents


| Path                        | Content                                                                                                                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/raw/sensor_traces.h5` | gzip-compressed e-nose traces, `float64`, shape (566 measurements, 13 channels, 189 time points). Datasets `row` and `code` match `metadata.csv`; `channel` names the 13 sensors |
| `data/raw/metadata.csv`     | English metadata and annotations, one row per measurement, row `i` matches trace `i`                                                                                             |
| `configs/`                  | shared protocol settings (`base.yaml`) and one file per task                                                                                                                     |
| `src/`                      | pipeline: loading, splits, models, metrics, tables                                                                                                                               |
| `scripts/`                  | entry points: full grid, table rebuild                                                                                                                                           |




## Dataset card



### Data collection

- **Event.** The data were collected during an official wine quality evaluation event run
under the standards of the International Organisation of Vine and Wine (OIV). The e-nose
measured the same samples that the sommelier panel judged.
- **Campaign.** There are six measurement days, 2025-09-08 to 2025-09-13, with 81 to 105
measurements per day. Each day is organised into shifts and sets (columns `shift` and
`set`). `startTime` gives the timing of every measurement.
- **Sensor array.** A multi-electrode lab-on-chip with 13 metal-oxide channels: 3 ZnO, 3 ZnO/In, 3 ZnO/Ni, 2 ZnO/Ce, and 2 ZnO/Ag. The sensors are heated to 200-400 C under closed-loop temperature control.
- **Protocol.** Each sample sits under a sealed glass dome. Headspace is drawn to the array at 3.3 mL/s, and resistance is sampled at 0.4 Hz. Each recording runs as follows: a 10 s baseline, 180 s of exposure from t = 10 s to t = 190 s, and recovery until t = 490 s. This gives 189 time points per channel. Clean air restores the baseline between samples.
- **Wines and repeats.** There are 566 measurements of 551 distinct wines. Fifteen wines
were measured twice.



### Annotations

- **Quality score.** A panel of professionals (wine producers, winery CEOs, PhD holders
in biological and technical sciences, critics and competition winners) scored the wines on
the 100-point Italian Sommelier Association (ISA) scale. The scale covers appearance, aroma,
taste and harmony. `wine_quality` is the mean score, with mean 82.3 and range 61.3 to 90.9.
Individual scores are kept in the 29 anonymised columns `R01`..`R29`. Each wine has 6 to 13
scores (median 11), and 2 wines have none.
- **Tasting notes.** For each wine, a single designated sommelier wrote a note using the ISA
vocabulary. Three text fields hold the notes:
  - `digest_description`: colour, nose, palate and food pairing (515 rows);
  - `good_description`: a one-line verdict of the wine's merits (499 rows);
  - `tech_description`: the producer's technical sheet (wine type, alcohol, vintage, grapes, sugar).
- **Aroma labels.** Four fields are derived from the notes. A plural name is a list; a singular name is one class.
  - `aroma_families_12`: a list from 12 families (Fruity, Floral, Herbaceous/vegetative, Spicy, Caramelized, Chemical, Wood, Nutty, Burned, Oxidized, Earthy, Microbiological);
  - `aroma_family_8`: one family from 8 of those (Fruity, Floral, Herbaceous/vegetative, Spicy, Caramelized, Burned, Earthy, Wood);
  - `aroma_descriptors_13`: a list from 13 primary descriptors (`cherry` through `grass`), or `[]`;
  - `aroma_cluster_6`: one of 6 clusters (`berries`, `tree_fruit`, `citrus`, `dried_fruit`, `fruit_unmentioned`, `bad`).



### Columns of `metadata.csv`


| Group         | Columns                                                                                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Identity      | `no`, `code` (wine index 1..551)                                                                                                                                        |
| Wine metadata | `name` (style), `year` (vintage), `raw` (grapes), `sugar` (g/dm3), `color`, `sweetness`, `strength_low`, `strength_high` (alcohol as a fraction), `strength_bins` (0-4) |
| Measurement   | `date`, `shift`, `set`, `number_of_samples`, `startTime`                                                                                                                |
| Quality       | `wine_quality`, `wine_quality_low_mean` (1 if the score is at or below 82.11)                                                                                           |
| Panel         | `R01`..`R29`                                                                                                                                                            |
| Text          | `tech_description`, `digest_description`, `good_description`                                                                                                            |
| Aroma         | `aroma_families_12`, `aroma_family_8`, `aroma_descriptors_13`, `aroma_cluster_6`, `aroma_labels_bad_wine_flags`                                                         |


Splits and deduplication use `code`.

### Preparation

1. Trace files were stacked into `sensor_traces.h5` in measurement order.
2. Wine metadata, panel scores and notes were joined to measurements by catalog code.
  Internal join and debug fields were dropped.
3. `color` and `sweetness` were derived from the style keywords in `name`.
  `strength_low` and `strength_high` were parsed from the strength field, and `strength_bins` are
   equal-frequency bins.
4. `scripts/translate_merged.py` builds `metadata.csv` from the Russian working copy. Free text is translated through the batch files in `data/translation/`. Styles, grape names and technical sheets go through a term glossary (`glossary.json`). Taster columns are renamed `R01`..`R29`.

Row order and every non-text column are unchanged. Each task yields the same labels from either file.

### Ethics and anonymisation

Sommelier identities are replaced by stable codes `R01`..`R29`, and the code-to-name key is
not distributed.

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

- **Translation.** The English text is a translation. The Russian originals are available
on request.



## Loading the data

```python
import h5py
import pandas as pd

with h5py.File("data/raw/sensor_traces.h5", "r") as f:
    X = f["traces"][:]                               # (566, 13, 189)
    # f["row"] and f["code"] match metadata.csv row order and wine id
meta = pd.read_csv("data/raw/metadata.csv")          # 566 rows, aligned with X
first = ~meta["code"].duplicated(keep="first")        # one measurement per wine
X, meta = X[first.values], meta[first].reset_index(drop=True)   # 551 wines
print(meta[["code", "date", "color", "wine_quality", "digest_description"]].head())
```



## Tasks


| Task         | Config                 | Target                               | Classes                                                          |
| ------------ | ---------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| Color-2      | `color2.yaml`          | `color`                              | red / white                                                      |
| Color-3      | `color3.yaml`          | `color`                              | red / white / rose                                               |
| ABV-b        | `abv_b.yaml`           | `strength_bins`                      | 5 alcohol bins                                                   |
| ABV          | `abv.yaml`             | `(strength_low + strength_high) / 2` | continuous alcohol (R2, MAE)                                     |
| Aroma        | `aroma_cluster_6.yaml` | `aroma_cluster_6`                    | berries, tree_fruit, citrus, dried_fruit, fruit_unmentioned, bad |
| Families-12  | `aroma_families_12.yaml` | `aroma_families_12`                | 12 families, multilabel                                          |
| Family-8     | `aroma_family_8.yaml`  | `aroma_family_8`                     | 8 families, one label                                            |
| Descriptors  | `aroma_descriptors_13.yaml` | `aroma_descriptors_13`          | 13 descriptors, multilabel                                       |
| Low-q        | `low_q.yaml`           | `wine_quality_bin(wine_quality, 0)`  | 1 at or below 82.11                                              |
| Low-q k=1..3 | `low_q_k{1,2,3}.yaml`  | `wine_quality_bin(wine_quality, k)`  | 1 at or below 76.73, 70.33, 65.63                                |




## Protocols

- **5x5 split by wine label (**`cv` **and** `holdout`**).**
  1. A 4:1:1 shuffle (seed 42) sets the first sixth of the rows aside as the outer test.
  2. StratifiedKFold(5) on the remaining train and validation rows gives 5 folds. ABV regression uses KFold(5) on the same outer split. If a binary task has fewer than 5 positives in train+val, fewer folds are used and the repeats increase so the cell still has 25 fits.
  3. Each fold is trained with 5 seeds, 25 fits in total.
  4. Each fit is scored twice: on its left-out fold (`cv`) and on the outer test (`holdout`). This gives 25 `cv` scores and 25 `holdout` scores per cell.

Naive and retrieval models use the same 25 runs. For retrieval, the training rows are the gallery and the evaluation rows are the queries.
- **Aroma folds.** `aroma_families_12`, `aroma_family_8`, `aroma_descriptors_13` and `aroma_cluster_6` share one split: the published outer 4:1:1 shuffle and `StratifiedKFold(5)` on `aroma_cluster_6`. The list targets do not draw their own folds. `aroma_family_8` removes its 47 blank cells only after that fold is fixed, so those wines leave training and scoring without moving any other wine. LODO is the same six days for every task.

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
- **Metrics.** Accuracy, F1, ROC-AUC and, for binary tasks, PR-AUC are reported as mean +/- population std over runs. F1 is binary for 2-class tasks and macro otherwise. ROC-AUC is one-vs-rest. ROC-AUC and PR-AUC are undefined for `svm`, which emits hard labels only. ABV regression reports R2 and MAE instead. Classification also reports macro and micro precision. For a single label, micro precision equals accuracy.
- **Aroma accuracy.** `aroma_cluster_6` and `aroma_family_8` score the argmax class. Accuracy is the fraction of evaluation rows whose predicted class equals the label. F1 is macro. ROC-AUC is one-vs-rest on the classes present in that evaluation set. `aroma_cluster_6` leaves PR-AUC empty. `aroma_family_8` fills PR-AUC with macro average precision. `subset_accuracy` stays empty for both.
- **Aroma lists.** `aroma_families_12` and `aroma_descriptors_13` are multilabel. A label scores positive at probability 0.5. `acc` is Hamming accuracy, the fraction of label bits that are correct. `subset_accuracy` is the exact match of the whole set. F1 and `precision_macro` are macro. `precision_micro` is micro. ROC-AUC and PR-AUC are macro over labels that have both classes in that evaluation set. An empty list is the all-negative label.



## Models

- Naive: stratified dummy classifier.
- Classical, on flattened traces: `rf`, `xgb`, `plsda`, `svm`. A list target fits one binary model per label. `svm` still emits hard labels.
- Neural: `MLSTMfcn`, `Transformer`, `CNNLSTM4`, `CNN1DClassifier`, `LSTMNet` and `EmissionCNN1DLSTM`. All are trained for 300 epochs with Adam (lr 1e-4), batch size 1000, and BCE on one-hot targets. A list target uses the indicator matrix and a sigmoid. ABV uses one output and MSE.
- Retrieval: `knn_pca` (kNN vote on 32 PCA components), `cosine_ssl` and `cosine_curvefit` (k = 1). ABV and the list targets predict the mean target of those same neighbours (`knn_pca` uses up to 13).



## Reproducing the paper tables

```bash
pip install -r requirements.txt
python scripts/run_all.py                 # full grid + tables (days on one CPU)
python scripts/write_tables.py            # rebuild tables from results/runs/
python scripts/run_all.py --tasks color2 --models rf knn_pca --drifts none --epochs 1 --repeats 1  # smoke test
```

The grid is 12 tasks x 14 models x {`lowess`, `no_lowess`} x {`none`, `snorm`, `tod_residual`, `day_residual`} x {5x5 split, LODO}. `aroma_families_12`, `aroma_family_8` and `aroma_descriptors_13` stay on `no_lowess` and drift `none`. Each run writes one file,
`results/runs/parts/{task}__{model}__{smooth}__{drift}/{split}__{run}.csv`, and
`run_all.py` skips runs whose file exists. Several copies can run at once: each run is
claimed with a lock file in `results/runs/claims/`. 


| Step | File                     | Role                                                                                                                  |
| ---- | ------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| 1    | `src/step1_load_data.py` | load traces, labels, days and start times; z-score, LOWESS                                                            |
| 2    | `src/splits.py`          | 5x5 split by wine label with outer test, leave-one-day-out                                                            |
| 3    | `src/step3_train.py`     | drift correction (`drift.py`), then one fit per run (`classical.py`, `step2_torch.py` + `models.py`, `retrieval.py`)  |
| 4    | `src/step4_analyze.py`   | accuracy, F1 and ROC-AUC per run; complete cells go to `results/runs/{task}__{model}__{split}__{smooth}__{drift}.csv` |
| 5    | `src/step5_visualize.py` | aggregate runs into `results/tables/*.csv`                                                                            |


Every table in `results/tables/` exists for each evaluation split `{split}` in `cv`,
`holdout` and `lodo`:


| CSV                     | Content                                                                                                                                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `all_runs.csv`          | mean and population std of acc, F1, roc_auc, pr_auc, subset_accuracy, precision_macro, precision_micro, r2 and mae for every task, model, split, smoothing and drift                                          |
| `low_q_threshold.csv`   | Low-quality detection vs. quality threshold: k, score cutoff, positive rate, best neural model on the holdout split by PR-AUC, and that model's ROC-AUC and PR-AUC. k=4 has no positives and no model            |
| `naive_{split}.csv`     | Naive baseline in every smoothing x drift cell                                                                                                                                                                   |
| `classical_{split}.csv` | `rf`, `xgb`, `plsda`, `svm` in every cell                                                                                                                                                                        |
| `neural_{split}.csv`    | the six neural models in every cell                                                                                                                                                                              |
| `ir_{split}.csv`        | the three retrieval models in every cell                                                                                                                                                                         |
| `best_{split}.csv`      | best end-to-end (classical or neural) vs best retrieval model by accuracy, at three grains: `overall` (across all smoothing x drift cells), `cell` (per smoothing x drift) and `model` (best cell of each model) |
| `ir_wins.csv`           | metrics on which the best retrieval model beats the best end-to-end model, per split, `overall` and per cell                                                                                                     |


All model families here use the single protocol above. The numbers in the submitted PDF came
from earlier scripts, in which classical and retrieval models were scored on fold-test rows and
the LODO neural models used a shorter training schedule. As a result, some cells can differ
slightly from the PDF.

## License, maintenance, citation

- **License.** Code and data are released under the Apache 2 (`LICENSE`).
- **Maintenance.** Issues and corrections go through the GitHub repository. Releases are
versioned, and any change to the data comes with a changelog entry and a regenerated
`metadata.csv`.
- **Citation:**

```bibtex
@inproceedings{winesmell2027,
  title     = {Developing Smell-Aware Intelligence with a Large Wine Olfaction Dataset},
  author    = {TODO},
  booktitle = {Proceedings of the 49th European Conference on Information Retrieval (ECIR)},
  year      = {2027}
}
```

