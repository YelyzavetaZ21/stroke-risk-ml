# Stroke prediction

Classification of stroke cases from patient records (`LogisticRegression`, `RandomForestClassifier`, `SVC`). The primary metric is recall for the stroke class. This is an educational project and **not a medical tool**.

## Contents

* `notebooks/01-eda.ipynb`: exploratory analysis, preprocessing, models, cross-validation, overfitting checks, feature importance, calibration, threshold selection, final model choice, limitations and references.
* Experiment tracking: [Weights & Biases project](https://forge.coreweave.com/wandb/zaecvolk21-21-personal-project/stroke-prediction)

## Project structure

```
.
├── app/
│   └── app.py                               (Streamlit app)
├── data/
│   └── healthcare-dataset-stroke-data.csv   (not included, see below)
├── models/
│   ├── lr_calibrated.joblib                 (calibrated model)
│   └── config.json                          (threshold, columns, age groups, BMI medians)
├── notebooks/
│   └── 01-eda.ipynb
└── README.md
```

## Data

[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle, author: fedesoriano): 5,110 patients and 12 columns. The file `healthcare-dataset-stroke-data.csv` is not included in this repository and has to be downloaded from Kaggle and placed in the `data/` folder. The test set contains 1,022 patients, 50 of them with a stroke; the training set contains 199 stroke cases.

## Main results

| Model | Test recall | Cross-validated recall |
|---|---|---|
| `LogisticRegression` + `class_weight='balanced'` | 0.80 | 0.794 ± 0.039 |
| `SVC` (linear kernel) | 0.78 | 0.814 ± 0.029 |
| `RandomForestClassifier` (best variant) | 0.16 | 0.211 ± 0.048 |

* ROC-AUC on the test set: `LogisticRegression` 0.843, `SVC` 0.845, `age` alone 0.834. The bootstrap difference between `LogisticRegression` and `age` alone is 0.009 (95% interval [-0.010, 0.028]).
* The predictive signal comes almost entirely from `age`.
* The probabilities of `LogisticRegression` with `class_weight='balanced'` were overestimated (Brier score 0.160). With sigmoid calibration the Brier score was 0.041 for both `LogisticRegression` and `SVC`.
* Working model: calibrated `LogisticRegression` with a threshold of 0.06, which gave a recall of 0.80 and a precision of 0.16 on the test set (40 true positives, 10 false negatives, 218 false positives).

## Limitations

The test set contains only 50 stroke cases, the dataset has no blood pressure, atrial fibrillation or diagnosis dates, and the model was not validated on external data. See the Limitations section of the notebook.

## Reproducing the results

1. Download the dataset from Kaggle (link above) and save it as `data/healthcare-dataset-stroke-data.csv`.
2. Install the libraries (tested with Python 3.11): `pandas` 3.0.5, `numpy` 2.4.6, `matplotlib` 3.11.1, `seaborn` 0.13.2, `scikit-learn` 1.9.0, `imbalanced-learn` 0.14.2, `wandb`, `jupyter`.
3. Open `notebooks/01-eda.ipynb` and choose `Kernel` > `Restart Kernel and Run All Cells`.
4. Logging to Weights & Biases is off by default (`LOG_TO_WANDB = False`). The calibrated model and the threshold of 0.06 are not logged there.

## Streamlit app

A small app (`app/app.py`) returns the score of the calibrated `LogisticRegression` for one patient and compares it with the working threshold of 0.06. It is a demonstration and **not a medical tool**.

```
streamlit run app/app.py
```

* The model and settings are loaded from `models/`. The notebook has to be run first (the last cells save `lr_calibrated.joblib` and `config.json`), or the two files have to be present.
* The saved model depends on the `scikit-learn` version (1.9.0 here) and may not load with another one.
* If BMI is left unknown, the app fills it with the median of the age group (`0-18`, `19-40`, `41-60`, `61-80`, `81+`) and sets `bmi_missing`. The medians were computed on the whole dataset before the train/test split, as in the notebook. In the training data a missing BMI was associated with a higher stroke rate (19.9% versus 4.3%), and the model uses this, so scores with unknown BMI should be treated with extra caution.
* The output of the app was checked against the notebook on one test patient: 0.048 with a known BMI and 0.146 with an unknown BMI, both in agreement with `predict_proba` in the notebook.
