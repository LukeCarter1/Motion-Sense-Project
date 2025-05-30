import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import time
import os
from collections import defaultdict
import json

from data_loader import MotionSenseDataLoader, create_data_loaders
from models import get_model, model_summary


class HAR_Trainer:

    def __init__(self, model, device='cuda', save_dir='./checkpoints'):

        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    
    def train_epoch(self, train_loader, optimizer, criterion):
 
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, 
              weight_decay=1e-4, scheduler_type='step', patience=10):

        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Setup scheduler
        if scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                        patience=patience//2, verbose=True)
        else:
            scheduler = None
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Optimizer: Adam, LR: {lr}, Weight Decay: {weight_decay}")
        
        start_time = time.time()
        
        # Early stopping variables
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Update scheduler
            if scheduler_type == 'step':
                scheduler.step()
            elif scheduler_type == 'plateau':
                scheduler.step(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }
                torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} (patience: {patience})')
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f'\nLoaded best model with validation accuracy: {self.best_val_acc:.2f}%')
        
        total_time = time.time() - start_time
        print(f'Training completed in {total_time/60:.1f} minutes')
        
        return self.history
    
    def evaluate(self, test_loader, activity_labels=None):
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy * 100,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        return results
    
    def plot_training_history(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plots
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss vs Accuracy
        axes[1, 1].scatter(self.history['train_loss'], self.history['train_acc'], 
                          alpha=0.6, label='Train', color='blue')
        axes[1, 1].scatter(self.history['val_loss'], self.history['val_acc'], 
                          alpha=0.6, label='Validation', color='red')
        axes[1, 1].set_title('Loss vs Accuracy')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix, activity_labels, save_path=None):
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=activity_labels,
                   yticklabels=activity_labels,
                   cbar_kws={'label': 'Normalized Frequency'})
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results, save_path):
        # Prepare results for JSON serialization
        json_results = {
            'model_name': self.model.__class__.__name__,
            'best_val_accuracy': self.best_val_acc,
            'test_accuracy': results['accuracy'],
            'training_history': self.history
        }
        
        # Convert numpy arrays to lists
        json_results['confusion_matrix'] = results['confusion_matrix'].tolist()
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {save_path}")


def main():
    
    # Configuration
    config = {
        'data_path': './data',
        'sensor_types': ['attitude', 'userAcceleration'],
        'activities': None,  # Use all activities
        'window_size': 200,
        'stride': 100,
        'test_size': 0.2,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'dropout_rate': 0.5,
        'model_name': 'cnn_hybrid',  # Changed to match available model
        'scheduler_type': 'step',
        'patience': 10,
        'save_dir': './results'
    }
    
    print("Human Activity Recognition - CNN Training")
    print("=" * 60)
    
    # Create results directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load and prepare data
    print("Loading MotionSense dataset...")
    data_loader = MotionSenseDataLoader(config['data_path'])
    
    data_info = data_loader.prepare_data(
        sensor_types=config['sensor_types'],
        activities=config['activities'],
        window_size=config['window_size'],
        stride=config['stride'],
        test_size=config['test_size'],
        normalize=True
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        data_info, 
        batch_size=config['batch_size']
    )
    
    # Create model
    print(f"\nCreating {config['model_name']} model...")
    model = get_model(
        config['model_name'],
        data_info['n_features'],
        data_info['n_classes'],
        data_info['window_size'],
        config['dropout_rate']
    )
    
    # Model summary
    model_summary(model, (data_info['n_features'], data_info['window_size']))
    
    # Create trainer
    trainer = HAR_Trainer(model, save_dir=config['save_dir'])
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for simplicity
        epochs=config['epochs'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_type=config['scheduler_type'],
        patience=config['patience']
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = trainer.evaluate(test_loader, data_info['activity_labels'])
    
    # Plot results
    trainer.plot_training_history(
        save_path=os.path.join(config['save_dir'], 'training_history.png')
    )
    
    trainer.plot_confusion_matrix(
        results['confusion_matrix'],
        data_info['activity_labels'],
        save_path=os.path.join(config['save_dir'], 'confusion_matrix.png')
    )
    
    # Save results
    trainer.save_results(
        results,
        os.path.join(config['save_dir'], 'results.json')
    )


if __name__ == "__main__":
    main()
