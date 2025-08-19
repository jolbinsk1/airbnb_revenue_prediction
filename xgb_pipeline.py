
import pandas as pd
import numpy as np
import json, zipfile, logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # necessary to use IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from analysis import analyze_feature_importance, plot_learning_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def baseline():

    """
    Train an XGBoost regression pipeline on the Airbnb dataset.
    
    Returns:
    --------
    best_model : Fitted pipeline with preprocessing & XGBRegressor.
    grid_search : Grid search object if tuning was run, else None
    
    """
    logging.info("Reading train and test files")
    train = pd.read_json("train.json", orient='records')
    test = pd.read_json("test.json", orient='records')
    seed = 123

    # Split train into train and validation
    train, valid = train_test_split(train, test_size=1/3, random_state=seed)
    
    # Create preprocessing pipeline
    preprocess = ColumnTransformer(
        transformers=[
            # Numerical features - impute then scale
            ("numerical", Pipeline(steps=[
                ('imputer', IterativeImputer(random_state=seed, max_iter=10)),
                ('scaler', StandardScaler())
            ]), ["lat", "lon", "bathrooms", "rooms", "guests", "num_reviews", "rating", "min_nights"]),
            
           # Categorical features - impute then one-hot encode
            ("categorical", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), ["room_type", "cancellation"]),
         ],
        remainder='drop'
    )

    label = 'revenue'

    xgb_regressor = Pipeline(steps=[
        ('preprocess', preprocess),
        ('xgb', XGBRegressor(
            n_estimators=300,
            max_depth=3,
            min_child_weight=5,
            learning_rate=0.1,
            reg_alpha=2,
            reg_lambda=3,
            gamma=1,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=seed,
            n_jobs=-1,
            eval_metric='mae'
        ))
    ])

    X_train = train.drop([label], axis=1)
    y_train = np.log1p(train[label].values)
    
    # Optional: Uncomment this block to perform grid search tuning
    # (Warning: VERY computationally expensive)
    
    '''
    logging.info("Starting grid search for hyperparameter tuning...")
    
    xgb_regressor = Pipeline(steps=[
        ('preprocess', preprocess),
        ('xgb', XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            eval_metric='mae'
        ))
    ])

    param_grid = {
        'xgb__max_depth': [3, 5],
        'xgb__min_child_weight': [3, 5, 7],
        'xgb__gamma': [0, 1],
        'xgb__reg_alpha': [0, 1, 2],
        'xgb__reg_lambda': [1, 3, 5],
        'xgb__subsample': [0.7, 0.8],
        'xgb__colsample_bytree': [0.7, 0.8],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__n_estimators': [300, 400, 500]
    }

    grid_search = GridSearchCV(
        xgb_regressor, 
        param_grid, 
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best CV score (neg MAE): {grid_search.best_score_:.4f}')

    '''

    # Comment out if running grid search
    xgb_regressor.fit(X_train, y_train)
    best_model = xgb_regressor  
    grid_search = None

    logging.info("Evaluating model performance...")
    
    for split_name, split in [("Train", train), ("Valid", valid)]:
        X_split = split.drop([label], axis=1)
        y_true = split[label].values
        
        # Predict and reverse log transform
        y_pred_log = best_model.predict(X_split)
        y_pred = np.expm1(y_pred_log)
        
        mae = mean_absolute_error(y_true, y_pred)
        
        logging.info(f"{split_name:>5} - MAE: {mae:.3f}")

    # Note: The professor provided a separate test set with hidden labels. 
    # This block is left here to demonstrate how to write predictions into the test.json file
    
    '''
    logging.info("Generating test predictions...")
    pred_test_log = best_model.predict(test)
    pred_test = np.expm1(pred_test_log)
    
    # Ensure no negative predictions
    pred_test = np.maximum(pred_test, 0)
    
    test[label] = pred_test
    predicted = test[['revenue']].to_dict(orient='records')

    # Save predictions
    logging.info("Saving predictions to baseline.zip...")
    with zipfile.ZipFile("baseline.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("redicted.json", json.dumps(predicted, indent=2))
    
    logging.info("Pipeline completed successfully!")
    '''
    
    return best_model, grid_search, train, valid

if __name__ == '__main__':
    
    try:
        best_model, grid_search, train, valid = baseline()
        
        # Analyze feature importance
        numeric_features = ["lat", "lon", "bathrooms", "rooms", "guests", "num_reviews", "rating", "min_nights"]
        categorical_features = ["room_type", "cancellation"]
        
        analyze_feature_importance(best_model, numeric_features, categorical_features)
        
        # plot learning curves        
        plot_learning_curves(best_model, train.drop(['revenue'], axis=1), np.log1p(train['revenue'].values))
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise