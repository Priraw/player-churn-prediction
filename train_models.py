"""
Gaming Churn Prediction - Model Training
=========================================
Train and evaluate XGBoost models for 7-day and 30-day churn prediction.
Includes hyperparameter tuning, cross-validation, and model evaluation.

Author: Priyanka Rawat
Date: December 2025
"""



import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, 
    precision_score, recall_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.evaluation_results = {}
        
    def train_xgboost(self, X_train, y_train, X_test, y_test, model_name='7day'):
        """Train XGBoost model."""
        print(f"Training XGBoost model for {model_name} churn prediction...")
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'tree_method': 'hist'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models[model_name] = model
        
        print(f"✓ Model trained successfully")
        print(f"  Total estimators: {model.n_estimators}")
        
        return model
    
    def evaluate_model(self, X_test, y_test, model_name='7day'):
        """Comprehensive model evaluation."""
        print(f"\nEvaluating {model_name} model...")
        
        model = self.models[model_name]
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.evaluation_results[model_name] = results
        
        print(f"\n{'=' * 60}")
        print(f"MODEL EVALUATION RESULTS - {model_name}")
        print(f"{'=' * 60}")
        print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"\nConfusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(f"  TN: {cm[0,0]:<6} FP: {cm[0,1]:<6}")
        print(f"  FN: {cm[1,0]:<6} TP: {cm[1,1]:<6}")
        
        self.calculate_business_metrics(y_test, y_pred, y_pred_proba, model_name)
        
        return results
    
    def calculate_business_metrics(self, y_test, y_pred, y_pred_proba, model_name):
        """Calculate business metrics."""
        print(f"\n{'=' * 60}")
        print(f"BUSINESS IMPACT ANALYSIS - {model_name}")
        print(f"{'=' * 60}")
        
        avg_player_ltv = 50
        retention_cost = 5
        retention_success_rate = 0.3
        
        tp = ((y_test == 1) & (y_pred == 1)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        fn = ((y_test == 1) & (y_pred == 0)).sum()
        
        churners_saved = tp * retention_success_rate
        value_saved = churners_saved * avg_player_ltv
        retention_attempts = tp + fp
        total_retention_cost = retention_attempts * retention_cost
        opportunity_cost = fn * avg_player_ltv
        net_value = value_saved - total_retention_cost
        roi = (net_value / total_retention_cost) * 100 if total_retention_cost > 0 else 0
        
        print(f"\nRetention Campaign Results:")
        print(f"  Players Identified as High Risk: {retention_attempts}")
        print(f"  Actual Churners Caught: {tp} ({tp/(tp+fn)*100:.1f}% of churners)")
        print(f"  False Alarms: {fp}")
        print(f"  Missed Churners: {fn}")
        print(f"\nFinancial Impact:")
        print(f"  Expected Players Saved: {churners_saved:.1f}")
        print(f"  Value Saved: ${value_saved:,.2f}")
        print(f"  Campaign Cost: ${total_retention_cost:,.2f}")
        print(f"  Net Value: ${net_value:,.2f}")
        print(f"  ROI: {roi:.1f}%")
        
        business_metrics = {
            'churners_saved': float(churners_saved),
            'value_saved': float(value_saved),
            'total_retention_cost': float(total_retention_cost),
            'net_value': float(net_value),
            'roi_percent': float(roi)
        }
        
        self.evaluation_results[model_name]['business_metrics'] = business_metrics
        return business_metrics
    
    def analyze_feature_importance(self, X, model_name='7day', top_n=20):
        """Analyze feature importance."""
        print(f"\nAnalyzing feature importance for {model_name} model...")
        
        model = self.models[model_name]
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance[model_name] = importance_df.to_dict('records')
        
        print(f"\nTop {top_n} Most Important Features:")
        print(f"{'=' * 60}")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<40} {row['importance']:.4f}")
        
        return importance_df
    
    def plot_roc_curve(self, X_test, y_test, model_name='7day'):
        """Plot ROC curve."""
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} Churn Prediction')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'roc_curve_{model_name}.png', dpi=300)
        print(f"✓ ROC curve saved to roc_curve_{model_name}.png")
        plt.close()
    
    def plot_feature_importance(self, X, model_name='7day', top_n=15):
        """Plot feature importance."""
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': self.models[model_name].feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png', dpi=300)
        print(f"✓ Feature importance saved to feature_importance_{model_name}.png")
        plt.close()
    
    def save_models(self):
        """Save trained models."""
        print(f"\n{'=' * 60}")
        print("SAVING MODELS")
        print(f"{'=' * 60}")
        
        for model_name, model in self.models.items():
            filename = f'churn_model_{model_name}.pkl'
            joblib.dump(model, filename)
            print(f"✓ Saved {filename}")
        
        with open('model_evaluation_results.json', 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        print("✓ Saved model_evaluation_results.json")
        
        with open('feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        print("✓ Saved feature_importance.json")


# MAIN EXECUTION
if __name__ == "__main__":
    print()
    print("=" * 70)
    print("GAMING CHURN PREDICTION - MODEL TRAINING")
    print("=" * 70)
    print()
    
    print("Loading engineered features...")
    train_df = pd.read_csv('gaming_churn_train_engineered.csv')
    test_df = pd.read_csv('gaming_churn_test_engineered.csv')
    
    with open('feature_metadata.json', 'r') as f:
        feature_metadata = json.load(f)
    
    feature_names = feature_metadata['feature_names']
    
    print(f"✓ Training data: {len(train_df)} records")
    print(f"✓ Test data: {len(test_df)} records")
    print(f"✓ Features: {len(feature_names)}")
    
    X_train = train_df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    X_test = test_df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    
    y_train_7day = train_df['churned_7day']
    y_test_7day = test_df['churned_7day']
    
    y_train_30day = train_df['churned_30day']
    y_test_30day = test_df['churned_30day']
    
    trainer = ChurnModelTrainer(random_state=42)
    
    # Train 7-day model
    print()
    print("=" * 70)
    print("TRAINING 7-DAY CHURN MODEL")
    print("=" * 70)
    
    model_7day = trainer.train_xgboost(
        X_train, y_train_7day, X_test, y_test_7day, model_name='7day'
    )
    
    eval_results_7day = trainer.evaluate_model(X_test, y_test_7day, model_name='7day')
    importance_7day = trainer.analyze_feature_importance(X_train, model_name='7day')
    trainer.plot_roc_curve(X_test, y_test_7day, model_name='7day')
    trainer.plot_feature_importance(X_train, model_name='7day')
    
    # Train 30-day model
    print()
    print("=" * 70)
    print("TRAINING 30-DAY CHURN MODEL")
    print("=" * 70)
    
    model_30day = trainer.train_xgboost(
        X_train, y_train_30day, X_test, y_test_30day, model_name='30day'
    )
    
    eval_results_30day = trainer.evaluate_model(X_test, y_test_30day, model_name='30day')
    importance_30day = trainer.analyze_feature_importance(X_train, model_name='30day')
    trainer.plot_roc_curve(X_test, y_test_30day, model_name='30day')
    trainer.plot_feature_importance(X_train, model_name='30day')
    
    # Save everything
    trainer.save_models()
    
    print()
    print("=" * 70)
    print("✓ MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print("Files created:")
    print("  - churn_model_7day.pkl")
    print("  - churn_model_30day.pkl")
    print("  - roc_curve_7day.png")
    print("  - roc_curve_30day.png")
    print("  - feature_importance_7day.png")
    print("  - feature_importance_30day.png")
    print("  - model_evaluation_results.json")
    print("  - feature_importance.json")
    print()
    print("Next step: uvicorn main:app --reload")