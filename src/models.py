import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class TransformerNLI:    
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Device selection: CUDA > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
            
        self.model.to(self.device)
        
    def tokenize_data(self, premises, hypotheses, labels=None, max_length=128):
        #Tokenize premise-hypothesis pairs
        encodings = self.tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        if labels is not None:
            encodings['labels'] = torch.tensor(labels)
            
        return encodings
    
    def create_dataset(self, encodings):
        #Create PyTorch dataset
        class NLIDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
                
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        return NLIDataset(encodings)
    
    def get_training_args(self, output_dir='../models/saved_models', 
                         num_epochs=3, batch_size=16, learning_rate=2e-5):
        
        # Use 'mps' if available for use_mps_device parameter
        use_mps = False  # Disabled for Colab
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='../results/logs',
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            save_total_limit=2,
            # use_mps_device disabled for Colab  # Enable MPS for Apple Silicon
        )
    
    def compute_metrics(self, pred):
        #Compute accuracy during training
        from sklearn.metrics import accuracy_score, f1_score
        
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        
        return {'accuracy': acc, 'f1': f1}
    
    def predict(self, premises, hypotheses):
        self.model.eval()
        encodings = self.tokenize_data(premises, hypotheses)
        
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        return predictions.cpu().numpy()
