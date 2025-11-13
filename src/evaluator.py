import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, Tuple

class ModelEvaluator:
    # Evaluate and analyze model performance
    def __init__(self, label_map=None):
        if label_map is None:
            self.label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        else:
            self.label_map = label_map

    def compute_metrics(self, y_true, y_pred) -> Dict:
        # Compute comprehensive metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        metrics = {
            'accuracy': accuracy,
            'macro_f1': f1.mean(),
            'per_class': {}
        }
        for i, label_name in self.label_map.items():
            metrics['per_class'][label_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, normalize=False):
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.3f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=list(self.label_map.values()),
            yticklabels=list(self.label_map.values())
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_classification_report(self, y_true, y_pred):
        #Print detailed classification report
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(
            y_true,
            y_pred,
            target_names=list(self.label_map.values())
        ))

    def analyze_errors(self, df, y_true, y_pred) -> pd.DataFrame:
        #Analyze prediction errors
        df_copy = df.copy()
        df_copy['predicted'] = y_pred
        df_copy['correct'] = y_true == y_pred
        errors = df_copy[~df_copy['correct']].copy()
        errors['error_type'] = errors.apply(
            lambda row: f"{self.label_map[row['label']]} → {self.label_map[row['predicted']]}",
            axis=1
        )
        return errors

    def analyze_by_length(self, df, y_true, y_pred) -> Tuple[pd.Series, pd.Series]:
        #Analyze performance by text length
        df_copy = df.copy()
        df_copy['premise_length'] = df_copy['premise'].str.split().str.len()
        df_copy['hypothesis_length'] = df_copy['hypothesis'].str.split().str.len()
        df_copy['correct'] = y_true == y_pred
        premise_bins = pd.cut(df_copy['premise_length'], bins=5)
        accuracy_by_premise = df_copy.groupby(premise_bins)['correct'].mean()
        hypothesis_bins = pd.cut(df_copy['hypothesis_length'], bins=5)
        accuracy_by_hypothesis = df_copy.groupby(hypothesis_bins)['correct'].mean()
        return accuracy_by_premise, accuracy_by_hypothesis