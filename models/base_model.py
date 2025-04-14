import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class TextCommandClassifier(nn.Module):
    """Base model for text command classification using BERT."""
    
    def __init__(self, num_labels, model_name='bert-base-uncased'):
        super(TextCommandClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get classification logits
        logits = self.classifier(pooled_output)
        
        return logits