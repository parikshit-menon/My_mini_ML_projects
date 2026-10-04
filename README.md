# My Mini ML Projects

This repository contains small machine-learning projects built to understand model implementation, data preparation, training, evaluation, and deployment.

## Projects

### 1. House Price Predictor

Located in [`House_price_predictor/`](House_price_predictor/), this project implements linear regression from scratch with NumPy.

It includes:

- A single-feature example using house size and price.
- A multiple-feature example using size, bedrooms, age, and distance to the city.
- Batch gradient descent.
- A normal-equation solution using the pseudoinverse.
- Training-loss and prediction experiments in Jupyter notebooks.

See the [House Price Predictor README](House_price_predictor/README.md) and [report](House_price_predictor/REPORT.md).

### 2. Air Quality Classifier

Located in [`Air_Quality_Classifier/`](Air_Quality_Classifier/), this project predicts an air-quality category from city and pollutant measurements.

It includes:

- A custom multiclass softmax-regression model.
- Numeric imputation and standardization.
- City imputation and one-hot encoding.
- A Streamlit prediction app.
- Saved preprocessing and model artifacts.

See the [Air Quality Classifier README](Air_Quality_Classifier/README.md) and [report](Air_Quality_Classifier/REPORT.md).

## Repository structure

```text
My_mini_ML_projects/
├── House_price_predictor/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   ├── main.py
│   ├── README.md
│   └── REPORT.md
├── Air_Quality_Classifier/
│   ├── data/
│   ├── notebooks/
│   ├── AQC_model.py
│   ├── app.py
│   ├── aqc_artifacts.joblib
│   ├── README.md
│   └── REPORT.md
└── README.md
```

## Requirements

The projects use Python and commonly used scientific Python tools, including:

- NumPy
- pandas
- Matplotlib
- Jupyter Notebook
- scikit-learn
- Streamlit
- joblib

Create and activate a virtual environment, then install the packages required by the project files.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib jupyter scikit-learn streamlit joblib
```

## Running the projects

### House price predictor

Open either notebook:

```powershell
jupyter notebook House_price_predictor/notebooks/Linear_R.ipynb
jupyter notebook House_price_predictor/notebooks/Linear_R_multiple.ipynb
```

The script version can be run from inside the project directory:

```powershell
cd House_price_predictor
python main.py
```

### Air-quality classifier

Run the Streamlit app from inside its project directory:

```powershell
cd Air_Quality_Classifier
streamlit run app.py
```

The app uses the saved `aqc_artifacts.joblib` file and displays an estimated AQI category with class probabilities.

## Results summary

The house-price notebooks report a pseudoinverse training MSE of `0.2000244603436692` for the single-feature dataset. For the multiple-feature dataset, the notebook reports a training MSE of `269856336.3497122` and a held-out test MSE of `1483002759.3997653`.

The air-quality repository contains data inspection and preprocessing outputs, but no completed model-evaluation output with accuracy, precision, recall, F1, or a confusion matrix. No air-quality performance metric is claimed here.

## Important limitations

These projects are educational implementations rather than production systems. The house-price data are small and synthetic. The air-quality data contain substantial missingness and imbalanced target classes. The Streamlit app uses placeholder default pollutant values, and its predictions should not be treated as official air-quality readings.

For stronger experiments, add reproducible evaluation scripts, cross-validation or time-aware validation, confusion matrices, per-class metrics, dependency pinning, and documented random seeds.
