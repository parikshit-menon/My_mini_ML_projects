# House Price Predictor Report

## 1. Objective

The project estimates a numeric house price with linear regression implemented directly in NumPy. It contains two experiments: one predictor (`size → price`) and four predictors (`size_sq_m`, `bedrooms`, `age_years`, `distance_to_city_km → price_usd`).

## 2. Data

### Single-feature dataset

`data/houses.csv` contains 12 rows and two numeric columns: `size` and `price`. The notebook prints values ranging from size 1.0–3.0 and prices 42–103.

### Multiple-feature dataset

`data/houses1.csv` contains 25 rows and these columns:

| Column | Role |
|---|---|
| `size_sq_m` | predictor |
| `bedrooms` | predictor |
| `age_years` | predictor |
| `distance_to_city_km` | predictor |
| `price_usd` | target |

The notebook constructs `X` from the four predictor columns and `y` from `price_usd`.

## 3. Model implementation

`src/Linear_R.py` defines `LinearRegression`. It adds an intercept column, predicts with `X @ theta`, computes mean squared error, and trains with full-batch gradient descent. It also exposes `normal_equation_closed`, which explicitly uses an inverse, and `normal_equation`, which uses `np.linalg.pinv`.

## 4. Verified notebook results

### Single-feature notebook

- Gradient descent: learning rate `0.01`, 100 iterations.
- Predictions for sizes `[2.5, 3.5]`: `[87.17041285, 116.63240872]`.
- Reported final training MSE: `0.44471243662775706`.
- Pseudoinverse fit: `theta = [11.85507246, 30.2421574]`.
- Pseudoinverse training MSE: `0.2000244603436692`.

The notebook cell labelled “after 10000 iterations” prints the earlier `final_mse` variable rather than recomputing it after the 10,000-iteration training call. That printed value (`0.44471243662775706`) should therefore not be interpreted as a freshly evaluated 10,000-iteration result.

### Multiple-feature notebook

The unscaled 100-iteration gradient-descent run emits NumPy overflow warnings. The pseudoinverse fit reports:

- `theta = [-34914.10743664, 2791.56749027, 9239.76757536, -2844.58708861, 58.74713877]`.
- Training MSE: `269856336.3497122`.

After feature standardization and 10,000 iterations, the notebook reports training MSE `269866627.78572613`. The gradient-descent and normal-equation predictions differ by a maximum absolute amount of `246.22800637356704` in the comparison cell.

For the `random_state=42`, 20% holdout split, training-set scaling statistics are:

- mean: `[109.5, 3.25, 12.5, 8.96]`
- standard deviation: `[30.07906249, 0.99373035, 8.86284379, 4.4229402]`

The reported held-out results are training MSE `193443242.05103606` and test MSE `1483002759.3997653`.

## 5. Interpretation and limitations

The reported metrics are MSE values on a very small synthetic dataset and should not be treated as real-estate performance claims. The very large test-to-train MSE difference indicates unstable generalization on this split. The gradient-descent overflow warning in the unscaled run demonstrates that the raw feature/target scales are unsuitable for the chosen learning rate. The notebook’s later scaling experiment improves numerical behavior but still reports a high test MSE.

For future experiments, fit preprocessing on the training set only, evaluate RMSE and R² alongside MSE, use cross-validation or repeated splits, and use a numerically stable least-squares solver rather than explicitly forming a matrix inverse.
