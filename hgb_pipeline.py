import logging
import json
import zipfile

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hist_baseline():

    """
    Train a Histogram Gradient Boosting Regressor on the Airbnb dataset.
    
    Returns:
    --------
    best_model : Fitted pipeline with preprocessing & HistGradientBoostingRegressor.
    grid_search : Grid search object if tuning was run, else None
    
    """
    
    logging.info("Reading train and test files")
    train = pd.read_json("train.json", orient='records')
    test = pd.read_json("test.json", orient='records')
    seed = 123

    # Split train into train and validation
    train, valid = train_test_split(train, test_size=1/3, random_state=seed)
    
    # Define preprocessing pipeline
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

    hbg_regressor = Pipeline(steps=[
        ('preprocess', preprocess), 
        ('HGB', HistGradientBoostingRegressor(
            learning_rate=0.1,          
            max_iter=300,               
            max_depth=3,                 
            min_samples_leaf=30,       
            l2_regularization=1.0,      
            max_features=0.7,           
            validation_fraction=0.15,   
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed
        ))
    ])

    X_train = train.drop([label], axis=1)
    y_train = np.log1p(train[label].values)

    # Optional: Uncomment this block to perform grid search tuning
    # (Warning: VERY computationally expensive)
    
    '''
    logging.info("Starting grid search for hyperparameter tuning...")

        # Create pipeline with basic model (parameters will be set by grid search)
    hgb_regressor = Pipeline(steps=[
        ('preprocess', preprocess), 
        ('HGB', HistGradientBoostingRegressor(
            validation_fraction=0.15,   
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed
        ))
    ])
    
    # Define parameter grid
    param_grid = {
        'HGB__learning_rate': [0.01, 0.05, 0.1],
        'HGB__max_depth': [2, 3, 4],
        'HGB__l2_regularization': [0.01, 0.1, 1.0],
        'HGB__max_iter': [100, 200, 300],
        'HGB__min_samples_leaf': [10, 20, 30]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        hgb_regressor, 
        param_grid, 
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    X_train = train.drop([label], axis=1)
    y_train = np.log1p(train[label].values)
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best CV score (neg MAE): {grid_search.best_score_:.4f}')
    '''
    
    # Comment out if running grid search
    hbg_regressor.fit(X_train, y_train)
    best_model = hbg_regressor  
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
    # Make predictions on test set
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
        zipf.writestr("predicted.json", json.dumps(predicted, indent=2))
    
    logging.info("Pipeline completed successfully!")
    '''
    
    return best_model, grid_search

def permutation_feature_importance(pipeline_model, X, y, seed=123, n_repeats=10):
    """
    Compute permutation feature importance.
    
    """
    # First transform features
    preprocess = pipeline_model.named_steps['preprocess']
    X_transformed = preprocess.transform(X)
    
    # Run permutation importance on the fitted model
    model = pipeline_model.named_steps['HGB']
    perm_importance = permutation_importance(
        model, X_transformed, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    
    # Get feature names after transformation
    feature_names = preprocess.get_feature_names_out()
    
    # Create dataframe
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values("importance_mean", ascending=False)
    
    return importances

if __name__ == "__main__":
    best_model, _ = hist_baseline()

    valid = pd.read_json("train.json", orient="records")
    train, valid = train_test_split(valid, test_size=1/3, random_state=123)

    X_valid = valid.drop(["revenue"], axis=1)
    y_valid = np.log1p(valid["revenue"].values)

    fi = permutation_feature_importance(best_model, X_valid, y_valid)
    print(fi.head(10))
