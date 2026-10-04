# Air Quality Classifier Report

## 1. Objective

The project predicts the categorical `AQI_Bucket` label from city and pollutant measurements. It is an educational classifier; the Streamlit app explicitly presents predictions as estimates rather than official air-quality readings.

## 2. Dataset schema

`data/city_day.csv` contains 29,531 rows and 16 columns:

`City`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, and `AQI_Bucket`.

The notebook reports 13 numeric columns and three string columns. Missing values are substantial, especially `Xylene` (18,109 missing), `PM10` (11,140), `NH3` (10,328), and `Toluene` (8,041). The target has 4,681 missing values, which are removed before modeling.

After removing missing labels, 24,850 rows remain. The six target classes and counts are:

| Class | Count |
|---|---:|
| Moderate | 8,829 |
| Satisfactory | 8,224 |
| Poor | 2,781 |
| Very Poor | 2,337 |
| Good | 1,341 |
| Severe | 1,338 |

## 3. Features and preprocessing

The notebook uses 13 inputs: `City` plus the 12 pollutant columns from `PM2.5` through `Xylene`. It excludes `Date`, `AQI`, and `AQI_Bucket`. Numeric features use median imputation and standardization. `City` uses most-frequent imputation and one-hot encoding.

The split is `train_test_split(test_size=0.2, random_state=42, stratify=y)`, producing 19,880 training rows and 4,970 test rows. The transformed matrices have 38 columns for both train and test. The class mapping is `Good: 0`, `Moderate: 1`, `Poor: 2`, `Satisfactory: 3`, `Severe: 4`, and `Very Poor: 5`.

## 4. Model implementation

`AQC_model.py` defines `SoftmaxRegression` with defaults `learning_rate=0.1` and `iterations=1000`. It initializes zero weights and bias, computes numerically stabilized softmax probabilities, minimizes cross-entropy, supports optional class weights, and predicts with the maximum class probability.

`app.py` loads `preprocessor`, `model`, and `classes` from `aqc_artifacts.joblib`. It accepts five hard-coded city choices and the 12 pollutant inputs, then displays the estimated category and sorted class probabilities.

## 5. Evaluation status

The inspected repository contains data inspection and preprocessing outputs, but the available notebook ends after importing `SoftmaxRegression`; it does not contain a completed training/evaluation output with accuracy, precision, recall, F1, or a confusion matrix. The saved joblib artifact confirms that a preprocessor/model/classes bundle exists, but this inspection environment does not include the runtime dependencies needed to safely deserialize and evaluate it.

Accordingly, no model-performance metric is claimed here. The repository should add a reproducible evaluation section that reports test accuracy, macro and weighted precision/recall/F1, per-class support, and a confusion matrix.

## 6. Limitations

The target is imbalanced, with Moderate and Satisfactory much more common than Good and Severe. Accuracy alone could therefore be misleading. The data are time-indexed city observations, while the notebook uses a random stratified split; a time-aware or city-aware validation design would better test deployment generalization. Missingness is extensive, and the app’s default pollutant values are placeholders, so interactive predictions should not be interpreted as sensor-grade outputs.
