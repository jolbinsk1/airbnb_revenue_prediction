# Airbnb Revenue Prediction

This project implements an **XGBoost regression pipeline** to predict Airbnb listing revenue. It includes preprocessing, model training, evaluation, feature importance analysis, and learning curve visualization.

## Project Overview

The pipeline performs the following steps:

1. Load and split the Airbnb dataset into training and validation sets.
2. Preprocess features:

   * **Numerical features:** impute missing values (IterativeImputer) and scale (StandardScaler)
   * **Categorical features:** impute missing values (most frequent) and one-hot encode
3. Train an **XGBoost regressor** on the processed data.
4. Evaluate performance using **mean absolute error (MAE)**.
5. Analyze **feature importance** and visualize **learning curves**.

Optional hyperparameter tuning is implemented via `GridSearchCV`.

## Dataset

* `train.json` – Training data
* `test.json` – Test data (for generating predictions)

The target variable is **`revenue`**.

### Features

**Numerical:**
`lat`, `lon`, `bathrooms`, `rooms`, `guests`, `num_reviews`, `rating`, `min_nights`

**Categorical:**
`room_type`, `cancellation`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/airbnb-revenue-prediction.git
cd airbnb-revenue-prediction
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline from the command line:

```bash
python pipeline.py
```

* The script trains the model, evaluates it on training and validation data, analyzes feature importance, and plots learning curves.
* Optional: uncomment the grid search section in `pipeline.py` for hyperparameter tuning.

## Project Structure

```
airbnb-revenue-prediction/
│
├── pipeline.py           # Main pipeline script
├── analysis.py           # Feature importance & learning curve functions
├── train.json            # Training dataset
├── test.json             # Test dataset
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Results

* MAE is reported on both **training** and **validation** sets.
* Feature importance plots identify the most predictive features.
* Learning curves help assess overfitting or underfitting.

## Future Work

* Explore additional features from listings or host data.
* Experiment with **other regression models** (LightGBM, CatBoost).
* Deploy as a REST API for real-time revenue predictions.
