import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def analyze_feature_importance(best_model, numeric_features, categorical_features):
    
    try:
        # get fitted XGBRegressor
        model = best_model.named_steps['xgb']
        
        # feature names (numeric + one-hot encoded categorical)
        feature_names = list(numeric_features) + \
                        list(best_model.named_steps['preprocess']
                             .named_transformers_['categorical']
                             .get_feature_names_out(categorical_features))

        # importance values
        importances = model.feature_importances_

        # pair up names + values
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print("\nFeature Importance Rankings:")
        print("----------------------------------------")
        for i, row in importance_df.iterrows():
            print(f"{i+1:2}. {row['feature']:<25}: {row['importance']:.4f}")

        return importance_df
    
    except Exception as e:
        logging.warning(f"Could not analyze feature importance: {e}")


def plot_learning_curves(model, X_train, y_train):
    
    try:
        from sklearn.model_selection import learning_curve
        import matplotlib.pyplot as plt
        
        train_sizes, train_scores, valid_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_absolute_error'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', label='Training Score')
        plt.plot(train_sizes, -valid_scores.mean(axis=1), 'o-', label='Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Absolute Error')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except ImportError:
        logging.info("matplotlib not available - skipping learning curves")
    except Exception as e:
        logging.warning(f"Could not plot learning curves: {e}")
