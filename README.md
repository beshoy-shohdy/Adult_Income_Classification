# Adult Income Classification

Predicts whether an individual earns more than $50K per year using demographic and work-related attributes from the Adult (Census Income) dataset. The project covers data cleaning, feature engineering, class balancing, and a comparison of Random Forest, PCA + Random Forest, and XGBoost models.

## Project Structure

```
.
├── Notebook.ipynb   # Full analysis: cleaning, EDA, modeling, evaluation
├── adult11.csv      # Input dataset (required, not included in this repo)
└── README.md
```

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- imbalanced-learn
- xgboost

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost
```

## Usage

1. Place `adult11.csv` in the same directory as the notebook.
2. Open and run `Notebook.ipynb` from top to bottom with Jupyter.

The notebook reads the data with a relative path (`pd.read_csv(r"adult11.csv")`), so it must be run from the folder containing the CSV.

## Pipeline Overview

**1. Data loading and inspection.** The dataset has 48,842 rows and 15 columns. Missing values are encoded as `?` and are converted to `NaN`. Affected columns are `workclass`, `occupation`, and `native-country`.

**2. Cleaning.**
- Removes 52 duplicate rows.
- Drops `fnlwgt` (a survey sampling weight) and `education` (redundant with the numeric `education-num`).

**3. Feature engineering.**
- `native-country` is collapsed to `US` / `Non-US`.
- `workclass`, `marital-status`, `occupation`, and `relationship` are grouped into fewer, more general categories.
- `capital_net` = `capital-gain` − `capital-loss`. The original capital columns are dropped.
- `has_capital_activity` is a binary flag set to 1 when there is any capital gain or loss.

**4. Target.** `salary` is encoded as `0` (`<=50K`) and `1` (`>50K`). The raw data is about 23% positive.

**5. Balancing.** The majority class is randomly downsampled so the dataset has 20,000 rows: 10,448 positive and 9,552 negative.

**6. Preprocessing** (inside a scikit-learn `ColumnTransformer`, so it is fit only on training data):
- `StandardScaler` for `age`, `education-num`, and `hours-per-week`.
- Yeo-Johnson transform followed by scaling for `capital_net`.
- One-hot encoding (`drop="first"`) for categorical and binary columns.

**7. Train/test split.** 80/20 stratified split with `random_state=42`, giving 16,000 training and 4,000 test rows.

**8. Models.**
- Random Forest (100 trees).
- PCA (retaining 95% of variance, 8 components) followed by Random Forest.
- XGBoost with default-style settings (100 estimators, max depth 6, learning rate 0.1).
- XGBoost tuned with `GridSearchCV` over 48 parameter combinations, scored by F1.

Cross-validation (5-fold stratified) is run on the training set for the RF and XGBoost pipelines.

## Results

Evaluated on the 4,000-row held-out test set:

| Model                 | Accuracy | Precision (>50K) | Recall (>50K) | F1 (>50K) | ROC-AUC |
|-----------------------|----------|------------------|---------------|-----------|---------|
| XGBoost               | 0.85     | 0.83             | 0.90          | 0.86      | 0.9232  |
| XGBoost (Grid Search) | 0.85     | 0.82             | 0.90          | 0.86      | 0.9230  |
| Random Forest         | 0.81     | 0.80             | 0.85          | 0.82      | 0.8808  |
| PCA + Random Forest   | 0.80     | 0.78             | 0.85          | 0.82      | —       |

The baseline XGBoost model performs as well as the grid-searched version, so it is the recommended model. The grid search's best cross-validated F1 was 0.848, with `colsample_bytree=0.8`, `learning_rate=0.1`, `max_depth=5`, `n_estimators=200`, and `subsample=1.0`. PCA did not improve Random Forest, so it was not kept.

Top features by Random Forest importance are `age`, `education-num`, `hours-per-week`, marital status (`Single`, `Previously-Married`), and capital activity.

## Limitations

- **Balanced evaluation.** Metrics are computed on a downsampled, roughly 50/50 test set. Performance on the real population, where about 23% earn over $50K, will differ. Precision in particular will be lower at the true base rate.
- **Missing dataset.** `adult11.csv` is referenced but not included. Its exact source and any modifications to the standard UCI Adult file are not documented in the notebook.
- **Unused import.** `SMOTE`, `BorderlineSMOTE`, and `RandomUnderSampler` are imported but not used; balancing is done by manual downsampling.
- **Sensitive attributes.** `gender` and `race` are included as model inputs. Predictions based on these attributes should not be used for decisions about individuals without a fairness review.
- **Notebook artifacts.** Some cell outputs include a Windows temp path and a pandas deprecation warning. Some markdown contains typos (e.g. "Catergorical", "XG_Boast").

## License

Add a license here before distributing the project.
