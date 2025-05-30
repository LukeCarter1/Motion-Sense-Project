import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import gc
import psutil
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import MotionSenseDataLoader, create_data_loaders
from models import get_model, model_summary
from train import HAR_Trainer


def print_memory_usage(stage=""):
    mem = psutil.virtual_memory()

def generate_confusion_matrix_heatmap(y_true, y_pred, activity_labels, save_dir):
    # Use the actual activity labels from the label encoder
    # These should be the original activity names like 'dws', 'ups', etc.
    
    # Map activity codes to readable names
    activity_name_mapping = {
        'dws': 'Downstairs',
        'ups': 'Upstairs', 
        'wlk': 'Walking',
        'jog': 'Jogging',
        'sit': 'Sitting',
        'std': 'Standing'
    }
    
    # Convert activity labels to readable names
    readable_labels = []
    for label in activity_labels:
        label_str = str(label).lower()
        if label_str in activity_name_mapping:
            readable_labels.append(activity_name_mapping[label_str])
        else:
            # If not in mapping, use the original label but make it readable
            readable_labels.append(str(label).replace('_', ' ').title())

    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=readable_labels, yticklabels=readable_labels, ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)')
    ax1.set_xlabel('Predicted Activity')
    ax1.set_ylabel('True Activity')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Plot 2: Normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=readable_labels, yticklabels=readable_labels, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted Activity')
    ax2.set_ylabel('True Activity')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
    plt.close()
    
    return cm, cm_normalized





def generate_training_history_plots(history, save_dir):
    if not history:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot training and validation accuracy
    if 'train_acc' in history and 'val_acc' in history:
        epochs = range(1, len(history['train_acc']) + 1)
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_history.pdf'), bbox_inches='tight')
    plt.close()


def get_predictions_and_labels(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_predictions)


class FastMotionSenseLoader(MotionSenseDataLoader):
    
    def create_fast_splits(self, sensor_types=["attitude", "userAcceleration"], 
                          activities=None, window_size=150, stride=100,
                          max_samples_per_subject_activity=1000,
                          train_subjects=None, val_subjects=None, test_subjects=None,
                          normalize=True):
        
        if activities is None:
            activities = ['dws', 'ups', 'wlk', 'jog', 'sit', 'std']
        
        if train_subjects is None:
            train_subjects = list(range(1, 14))
        if val_subjects is None:
            val_subjects = list(range(15, 19))
        if test_subjects is None:
            test_subjects = list(range(20, 24))
        
        splits = {
            'train': {'subjects': train_subjects, 'X': [], 'y': []},
            'val': {'subjects': val_subjects, 'X': [], 'y': []},
            'test': {'subjects': test_subjects, 'X': [], 'y': []}
        }
        
        dt_list = self.set_data_types(sensor_types)
        
        for split_name, split_info in splits.items():
            for subject_id in split_info['subjects']:
                subject_data = []
                
                for act_id, act in enumerate(activities):
                    if act not in self.TRIAL_CODES:
                        continue
                    
                    activity_samples = 0
                    
                    for trial in self.TRIAL_CODES[act]:
                        if activity_samples >= max_samples_per_subject_activity:
                            break
                            
                        fname = os.path.join(
                            self.data_path, 
                            f"A_DeviceMotion_data/{act}_{trial}/sub_{subject_id}.csv"
                        )
                        
                        try:
                            raw_data = pd.read_csv(fname)
                            if 'Unnamed: 0' in raw_data.columns:
                                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                            
                            remaining_samples = max_samples_per_subject_activity - activity_samples
                            if len(raw_data) > remaining_samples:
                                step = len(raw_data) // remaining_samples
                                raw_data = raw_data.iloc[::max(1, step)].head(remaining_samples)
                            
                            sensor_data = raw_data[dt_list].values
                            
                            for sample in sensor_data:
                                row = list(sample) + [act_id]
                                subject_data.append(row)
                                activity_samples += 1
                                
                                if activity_samples >= max_samples_per_subject_activity:
                                    break
                                    
                        except FileNotFoundError:
                            continue
                        except Exception as e:
                            continue
                
                if len(subject_data) > 0:
                    columns = dt_list + ["activity"]
                    subject_df = pd.DataFrame(subject_data, columns=columns)
                    subject_df = subject_df.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    subject_X, subject_y = self.create_windows(subject_df, window_size, stride)
                    
                    if len(subject_X) > 0:
                        split_info['X'].append(subject_X)
                        split_info['y'].append(subject_y)
        
        final_splits = {}
        
        for split_name, split_info in splits.items():
            if len(split_info['X']) > 0:
                X_combined = np.concatenate(split_info['X'], axis=0)
                y_combined = np.concatenate(split_info['y'], axis=0)
                
                y_encoded = self.label_encoder.fit_transform(y_combined) if split_name == 'train' else self.label_encoder.transform(y_combined)
                
                final_splits[split_name] = {
                    'X': X_combined.astype(np.float32),
                    'y': y_encoded.astype(np.int64),
                    'subjects': split_info['subjects']
                }
            else:
                final_splits[split_name] = {'X': np.array([]), 'y': np.array([]), 'subjects': split_info['subjects']}
        
        if normalize and len(final_splits['train']['X']) > 0:
            train_X = final_splits['train']['X']
            n_train, n_features, n_timesteps = train_X.shape
            
            train_X_flat = train_X.reshape(n_train, -1)
            train_X_flat = self.scaler.fit_transform(train_X_flat)
            final_splits['train']['X'] = train_X_flat.reshape(n_train, n_features, n_timesteps)
            
            for split_name in ['val', 'test']:
                if len(final_splits[split_name]['X']) > 0:
                    split_X = final_splits[split_name]['X']
                    n_samples = split_X.shape[0]
                    
                    split_X_flat = split_X.reshape(n_samples, -1)
                    split_X_flat = self.scaler.transform(split_X_flat)
                    final_splits[split_name]['X'] = split_X_flat.reshape(n_samples, n_features, n_timesteps)
        
        data_info = {
            'X_train': final_splits['train']['X'],
            'X_val': final_splits['val']['X'],
            'X_test': final_splits['test']['X'],
            'y_train': final_splits['train']['y'],
            'y_val': final_splits['val']['y'],
            'y_test': final_splits['test']['y'],
            'n_classes': len(activities),
            'n_features': len(sensor_types) * 3,
            'window_size': window_size,
            'activity_labels': self.label_encoder.classes_,
            'sensor_types': sensor_types,
            'subject_splits': {
                'train': final_splits['train']['subjects'],
                'val': final_splits['val']['subjects'], 
                'test': final_splits['test']['subjects']
            }
        }
        
        return data_info


def create_3way_data_loaders(data_info, batch_size=32, num_workers=0):
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_info['X_train']),
        torch.LongTensor(data_info['y_train'])
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_info['X_val']),
        torch.LongTensor(data_info['y_val'])
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data_info['X_test']),
        torch.LongTensor(data_info['y_test'])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def main():
    config = {
        'data_path': './data',
        'sensor_types': ['attitude', 'userAcceleration'],
        'activities': ['dws', 'ups', 'wlk', 'jog', 'sit', 'std'],
        'window_size': 300,
        'stride': 75,
        'normalize': True,
        'max_samples_per_subject_activity': 800,
        
        'train_subjects': list(range(1, 14)),
        'val_subjects': list(range(15, 19)),
        'test_subjects': list(range(20, 24)),
        
        'batch_size': 16,
        'epochs':40,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'dropout_rate': 0.5,
        
        'model_name': 'cnn_hybrid',
        'scheduler_type': 'step',
        'patience': 6,
        
        'save_dir': f'./results',
        'save_plots': True,
        'verbose': False
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    try:
        data_loader = FastMotionSenseLoader(config['data_path'])
        
        data_info = data_loader.create_fast_splits(
            sensor_types=config['sensor_types'],
            activities=config['activities'],
            window_size=config['window_size'],
            stride=config['stride'],
            max_samples_per_subject_activity=config['max_samples_per_subject_activity'],
            train_subjects=config['train_subjects'],
            val_subjects=config['val_subjects'],
            test_subjects=config['test_subjects'],
            normalize=config['normalize']
        )
        
        train_size = data_info['X_train'].nbytes + data_info['y_train'].nbytes
        val_size = data_info['X_val'].nbytes + data_info['y_val'].nbytes
        test_size = data_info['X_test'].nbytes + data_info['y_test'].nbytes
        total_size_mb = (train_size + val_size + test_size) / (1024**2)
        
        data_summary = {
            'n_train_samples': int(len(data_info['X_train'])),
            'n_val_samples': int(len(data_info['X_val'])),
            'n_test_samples': int(len(data_info['X_test'])),
            'n_features': int(data_info['n_features']),
            'n_classes': int(data_info['n_classes']),
            'activity_labels': [str(label) for label in data_info['activity_labels']],
            'window_size': int(data_info['window_size']),
            'sensor_types': config['sensor_types'],
            'dataset_size_mb': float(total_size_mb),
            'subject_splits': {
                'train': [int(s) for s in data_info['subject_splits']['train']],
                'val': [int(s) for s in data_info['subject_splits']['val']],
                'test': [int(s) for s in data_info['subject_splits']['test']]
            },
            'fast_training': True,
            'max_samples_per_subject_activity': config['max_samples_per_subject_activity']
        }
        
        with open(os.path.join(config['save_dir'], 'data_summary.json'), 'w') as f:
            json.dump(data_summary, f, indent=2)
            
    except Exception as e:
        print(f"Error creating fast splits: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        train_loader, val_loader, test_loader = create_3way_data_loaders(
            data_info, 
            batch_size=config['batch_size'],
            num_workers=0
        )
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    gc.collect()
    
    try:
        model = get_model(
            config['model_name'],
            data_info['n_features'],
            data_info['n_classes'],
            data_info['window_size'],
            config['dropout_rate']
        )
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    start_time = datetime.now()
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        trainer = HAR_Trainer(model, device=device, save_dir=config['save_dir'])
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            scheduler_type=config['scheduler_type'],
            patience=config['patience']
        )
        
        training_time = datetime.now() - start_time
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    try:
        # Convert activity labels to strings to avoid sklearn issues
        activity_labels_str = [str(label) for label in data_info['activity_labels']]
        
        results = trainer.evaluate(test_loader, activity_labels_str)
        
        print(f"Test accuracy: {results['accuracy']:.2f}%")
        print(f"Training time: {datetime.now() - start_time}")
        
       
        # Get predictions for confusion matrix
        y_true, y_pred = get_predictions_and_labels(model, test_loader, device)
        
        # Generate confusion matrix with activity names
        cm, cm_normalized = generate_confusion_matrix_heatmap(
            y_true, y_pred, activity_labels_str, config['save_dir']
        )
        
        # Generate training history plots
        generate_training_history_plots(history, config['save_dir'])
        
        # Save detailed results with confusion matrix
        detailed_results = {
            **results,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'training_time_seconds': training_time.total_seconds(),
            'training_completed_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(config['save_dir'], 'detailed_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
    except Exception as e:
        print(f"Error during evaluation or visualization: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
