import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class FakeNewsClassifier:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def initialize_models(self):
        """
        Initialize multiple ML models
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_model(self, name, model, X_train, y_train):
        """
        Train a single model
        """
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        self.trained_models[name] = model
        print(f"{name} training complete!")
        return model
    
    def evaluate_model(self, name, model, X_test, y_test):
        """
        Evaluate a single model
        """
        print(f"\nEvaluating {name}...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
        
        self.results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if auc:
            print(f"  AUC:       {auc:.4f}")
        
        return self.results[name]
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models
        """
        if not self.models:
            self.initialize_models()
        
        for name, model in self.models.items():
            self.train_model(name, model, X_train, y_train)
            self.evaluate_model(name, model, X_test, y_test)
    
    def get_results_dataframe(self):
        """
        Get results as a dataframe for easy comparison
        """
        results_list = []
        for name, metrics in self.results.items():
            row = {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC': metrics['auc'] if metrics['auc'] else 0
            }
            results_list.append(row)
        
        return pd.DataFrame(results_list).sort_values('F1-Score', ascending=False)
    
    def plot_confusion_matrices(self, y_test, figsize=(15, 10)):
        """
        Plot confusion matrices for all models
        """
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.ravel() if n_models > 1 else [axes]
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = confusion_matrix(y_test, metrics['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{name}\nF1: {metrics["f1_score"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticklabels(['Fake', 'True'])
            axes[idx].set_yticklabels(['Fake', 'True'])
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_test):
        """
        Plot ROC curves for all models
        """
        plt.figure(figsize=(10, 8))
        
        for name, metrics in self.results.items():
            if metrics['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
                auc = metrics['auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_models(self, directory='../models/'):
        """
        Save all trained models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = name.replace(' ', '_').lower()
            filepath = f'{directory}{filename}.pkl'
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        """
        return joblib.load(filepath)


class EnsembleClassifier:
    def __init__(self, models_dict):
        """
        models_dict: dictionary of {name: trained_model}
        """
        self.models = models_dict
        self.weights = None
    
    def predict_proba(self, X):
        """
        Weighted voting for probability predictions.
        Ensures X is a DataFrame so sklearn can validate feature names.
        """
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        predictions = []

        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                else:
                    pred_proba = model.predict(X).astype(float)
                predictions.append(pred_proba)
            except Exception as e:
                print(f"Warning: {name} failed — {e}. Skipping.")

        if not predictions:
            raise ValueError("All models failed during prediction.")

        ensemble_proba = np.mean(predictions, axis=0)
        return ensemble_proba
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble performance
        """
        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics


if __name__ == "__main__":
    print("Model training module loaded successfully!")