import joblib
import os

def get_model_path():
    """
    Check which models are available and return the best one
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Priority order: smaller models first for deployment
    model_priority = [
        'logistic_regression.pkl',
        'random_forest.pkl',
        'xgboost.pkl',
        'gradient_boosting.pkl',
        'ensemble_model.pkl'
    ]
    
    for model_name in model_priority:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            # Check file size
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb < 100:  # Only use models under 100MB
                print(f"Using model: {model_name} ({size_mb:.2f} MB)")
                return model_path
    
    return None