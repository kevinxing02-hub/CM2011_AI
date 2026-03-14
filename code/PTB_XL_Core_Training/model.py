import torch.nn as nn

class LeadAgnosticTransformer(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=2, num_layers=2):
        super(LeadAgnosticTransformer, self).__init__()
        
        # 1. Feature Extractor (1D-CNN)
        # Input to CNN will be [Batch * Leads, 1, Time]
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1) # This compresses the Time axis to 1
        )
        
        # 2. Transformer Encoder
        # Input to Transformer will be [Batch, Leads, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Multi-label Classification Head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [Batch, Leads, Time]
        b, l, t = x.shape
        
        # FOLD: Combine Batch and Leads so CNN processes each lead individually
        x = x.reshape(b * l, 1, t) 
        
        # CNN: Extract features per lead
        x = self.cnn(x) # Output: [B*L, d_model, 1]
        
        # UNFOLD: Back to [Batch, Leads, Features]
        x = x.view(b, l, -1)
        
        # TRANSFORMER: "Attention" across the leads
        x = self.transformer(x)
        
        # GLOBAL POOLING: Mean across the lead dimension
        x = x.mean(dim=1) 
        
        # OUTPUT: Logits for multi-label classification
        return self.classifier(x)