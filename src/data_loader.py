from datasets import load_dataset
import pandas as pd
from typing import Tuple, Dict

class ANLIDataLoader:
    def __init__(self):
        self.label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Load ANLI splits
        print("Loading ANLI dataset:")
        dataset = load_dataset("facebook/anli")
        train_df = pd.DataFrame(dataset['train_r2'])
        dev_df = pd.DataFrame(dataset['dev_r2'])
        test_df = pd.DataFrame(dataset['test_r2'])
        print(f"Train size: {len(train_df)}")
        print(f"Dev size: {len(dev_df)}")
        print(f"Test size: {len(test_df)}")
        return train_df, dev_df, test_df

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        # Get basic statistics about the dataset
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_premise_length': df['premise'].str.split().str.len().mean(),
            'avg_hypothesis_length': df['hypothesis'].str.split().str.len().mean(),
            'unique_premises': df['premise'].nunique(),
            'unique_hypotheses': df['hypothesis'].nunique()
        }
        return stats

    def prepare_text_pairs(self, df: pd.DataFrame) -> Tuple[list, list]:
        # Prepare text pairs for model input
        premises = df['premise'].tolist()
        hypotheses = df['hypothesis'].tolist()
        return premises, hypotheses