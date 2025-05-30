
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Hybrid(nn.Module):
    
    def __init__(self, n_features, n_classes, window_size, dropout_rate=0.5):
        super(CNN_Hybrid, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.window_size = window_size
        
        # 1D convolution branch (temporal features)
        self.conv1d_1 = nn.Conv1d(n_features, 64, kernel_size=7, padding=3)
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool1d = nn.MaxPool1d(2)
        
        # 2D convolution branch (spatial-temporal features)
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3))
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.pool2d = nn.MaxPool2d((1, 2))
        
        # Batch normalization
        self.bn1d_1 = nn.BatchNorm1d(64)
        self.bn1d_2 = nn.BatchNorm1d(128)
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.bn2d_2 = nn.BatchNorm2d(64)
        
        # Calculate combined feature size
        combined_size = self._get_combined_output_size()
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(combined_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        
    def _get_combined_output_size(self):
        """Calculate combined output size from both branches"""
        x = torch.randn(1, self.n_features, self.window_size)
        
        # 1D branch
        x1d = self.pool1d(F.relu(self.conv1d_1(x)))
        x1d = self.pool1d(F.relu(self.conv1d_2(x1d)))
        x1d_flat = x1d.view(1, -1)
        
        # 2D branch
        x2d = x.unsqueeze(1)  # Add channel dimension
        x2d = self.pool2d(F.relu(self.conv2d_1(x2d)))
        x2d = self.pool2d(F.relu(self.conv2d_2(x2d)))
        x2d_flat = x2d.view(1, -1)
        
        return x1d_flat.size(1) + x2d_flat.size(1)
    
    def forward(self, x):

        batch_size = x.size(0)
        
        # 1D convolution branch
        x1d = self.pool1d(F.relu(self.bn1d_1(self.conv1d_1(x))))
        x1d = self.pool1d(F.relu(self.bn1d_2(self.conv1d_2(x1d))))
        x1d_flat = x1d.view(batch_size, -1)
        
        # 2D convolution branch
        x2d = x.unsqueeze(1)  # [batch, 1, n_features, window_size]
        x2d = self.pool2d(F.relu(self.bn2d_1(self.conv2d_1(x2d))))
        x2d = self.pool2d(F.relu(self.bn2d_2(self.conv2d_2(x2d))))
        x2d_flat = x2d.view(batch_size, -1)
        
        # Combine features
        x_combined = torch.cat([x1d_flat, x2d_flat], dim=1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x_combined)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def get_model(model_name, n_features, n_classes, window_size, dropout_rate=0.5):
    models = {
        'cnn_hybrid': CNN_Hybrid,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    return models[model_name](n_features, n_classes, window_size, dropout_rate)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_shape):
   
    print(f"\nModel: {model.__class__.__name__}")
    print("="*60)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test with dummy input
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        try:
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error in forward pass: {e}")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage and testing
    n_features = 6  # attitude + userAcceleration
    n_classes = 6   # 6 activities
    window_size = 200
    
    print("Testing CNN Hybrid Model for Human Activity Recognition")
    print("="*60)
    
    # Create and test the hybrid model
    model = CNN_Hybrid(n_features, n_classes, window_size)
    model_summary(model, (n_features, window_size))
    
    # Test forward pass with batch
    batch_size = 4
    dummy_batch = torch.randn(batch_size, n_features, window_size)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_batch)
        print(f"Batch test - Input: {dummy_batch.shape}, Output: {output.shape}")
        print(f"Output probabilities shape: {F.softmax(output, dim=1).shape}")
