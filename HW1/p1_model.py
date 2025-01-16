import random
from torch import nn

class Classifier(nn.Module):
    def __init__(self, backbone, in_features, n_class, hidden_size=512, dropout=0.1) -> None:
        super().__init__()
        # First layer
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        # Second layer
        self.fc2 = nn.Linear(hidden_size, n_class)
        
        self.backbone = backbone
        self.feature_vector = None  # To store the output feature vector of the first layer

    def forward(self, img):
        # Pass through backbone to get embeddings
        embeds = self.backbone(img)
        
        # Pass through the first layer and store the output
        features = self.fc1(embeds)
        self.feature_vector = features  # Save the feature vector
        
        # Pass through the second layer to get logits
        logits = self.fc2(features)
        
        return logits

    # Method to retrieve the stored feature vector
    def get_features(self):
        return self.feature_vector
