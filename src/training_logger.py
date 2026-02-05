"""
Training metrics logger for model performance monitoring
"""

import json
import numpy as np
from pathlib import Path


class TrainingLogger:
    """Logs training metrics for visualization"""
    
    def __init__(self, output_dir='models/training_history'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'xgboost': {
                'epochs': [],
                'train_accuracy': [],
                'train_loss': [],
                'val_accuracy': [],
                'val_loss': []
            },
            'lightgbm': {
                'epochs': [],
                'train_accuracy': [],
                'train_loss': [],
                'val_accuracy': [],
                'val_loss': []
            },
            'logistic': {
                'epochs': [],
                'train_accuracy': [],
                'train_loss': [],
                'val_accuracy': [],
                'val_loss': []
            }
        }
        
        self.model_metrics = {
            'xgboost': {},
            'lightgbm': {},
            'logistic': {}
        }
    
    def log_training_callback(self, model_name, epoch, 
                            train_acc, train_loss, 
                            val_acc=None, val_loss=None):
        """Log metrics for each epoch"""
        if model_name not in self.history:
            return
        
        self.history[model_name]['epochs'].append(epoch)
        self.history[model_name]['train_accuracy'].append(float(train_acc))
        self.history[model_name]['train_loss'].append(float(train_loss))
        
        if val_acc is not None:
            self.history[model_name]['val_accuracy'].append(float(val_acc))
        if val_loss is not None:
            self.history[model_name]['val_loss'].append(float(val_loss))
    
    def log_final_metrics(self, model_name, metrics):
        """Log final metrics for a model"""
        if model_name not in self.model_metrics:
            return
        
        self.model_metrics[model_name] = {
            'train_accuracy': float(metrics.get('train_acc', 0)),
            'train_loss': float(metrics.get('train_logloss', 0)),
            'val_accuracy': float(metrics.get('val_acc', 0)),
            'val_loss': float(metrics.get('val_logloss', 0))
        }
    
    def log_feature_importance(self, model_name, feature_importance, feature_names):
        """Log feature importance from tree-based models"""
        importance_dict = {}
        
        # Normalize importance
        if isinstance(feature_importance, np.ndarray):
            total = feature_importance.sum()
            if total > 0:
                feature_importance = feature_importance / total
            
            for name, importance in zip(feature_names, feature_importance):
                importance_dict[str(name)] = float(importance)
        
        if 'feature_importance' not in self.model_metrics[model_name]:
            self.model_metrics[model_name]['feature_importance'] = {}
        
        self.model_metrics[model_name]['feature_importance'] = importance_dict
    
    def save(self):
        """Save training history to JSON"""
        output_file = self.output_dir / 'training_history.json'
        
        data = {
            'training_curves': self.history,
            'model_metrics': self.model_metrics
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(output_file)
    
    @staticmethod
    def load(history_file='models/training_history/training_history.json'):
        """Load training history from JSON"""
        path = Path(history_file)
        
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data


def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    
    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0,
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'f1_score': float(2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn)))) 
                   if (tp + fp > 0 and tp + fn > 0) else 0
    }
