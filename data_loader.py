

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class MotionSenseDataLoader:
    def __init__(self, data_path="./data"):
        self.data_path = data_path
        self.ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
        self.TRIAL_CODES = {
            "dws": [1, 2, 11],
            "ups": [3, 4, 12], 
            "wlk": [7, 8, 15],
            "jog": [9, 16],
            "std": [6, 14],
            "sit": [5, 13]
        }
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def get_ds_infos(self):
        try:
            dss = pd.read_csv(os.path.join(self.data_path, "data_subjects_info.csv"))
            return dss
        except FileNotFoundError:
            # Create dummy subject info if file doesn't exist
            return pd.DataFrame({"code": range(1, 25)})
    
    def set_data_types(self, data_types=["attitude", "userAcceleration"]):
        dt_list = []
        for t in data_types:
            if t != "attitude":
                dt_list.extend([f"{t}.x", f"{t}.y", f"{t}.z"])
            else:
                dt_list.extend([f"{t}.roll", f"{t}.pitch", f"{t}.yaw"])
        return dt_list
    
    def create_time_series(self, sensor_types=["attitude", "userAcceleration"], 
                          activities=None, mode="raw"):
        if activities is None:
            activities = self.ACT_LABELS
            
        dt_list = self.set_data_types(sensor_types)
        num_sensors = len(sensor_types)
        num_features = num_sensors * 3 if mode == "raw" else num_sensors
        
        dataset = []
        ds_list = self.get_ds_infos()
        
        
        for sub_id in ds_list["code"]:
            for act_id, act in enumerate(activities):
                if act not in self.TRIAL_CODES:
                    continue
                    
                for trial in self.TRIAL_CODES[act]:
                    fname = os.path.join(
                        self.data_path, 
                        f"A_DeviceMotion_data/{act}_{trial}/sub_{int(sub_id)}.csv"
                    )
                    
                    try:
                        raw_data = pd.read_csv(fname)
                        if 'Unnamed: 0' in raw_data.columns:
                            raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                        
                        # Extract sensor data
                        if mode == "raw":
                            sensor_data = raw_data[dt_list].values
                        else:  # magnitude mode
                            sensor_data = np.zeros((len(raw_data), num_sensors))
                            for i, sensor_type in enumerate(sensor_types):
                                cols = self.set_data_types([sensor_type])
                                if sensor_type == "attitude":
                                    # For attitude, use euclidean norm
                                    sensor_data[:, i] = np.sqrt(
                                        (raw_data[cols] ** 2).sum(axis=1)
                                    )
                                else:
                                    # For other sensors, use magnitude
                                    sensor_data[:, i] = np.sqrt(
                                        (raw_data[cols] ** 2).sum(axis=1)
                                    )
                        
                        # Add labels
                        labels = np.full(len(raw_data), act_id)
                        
                        # Combine data
                        for i in range(len(sensor_data)):
                            row = list(sensor_data[i]) + [labels[i]]
                            dataset.append(row)
                            
                    except FileNotFoundError:
                        print(f"[WARNING] -- File not found: {fname}")
                        continue
        
        # Create DataFrame
        if mode == "raw":
            columns = dt_list + ["activity"]
        else:
            columns = [f"{sensor}_mag" for sensor in sensor_types] + ["activity"]
            
        dataset_df = pd.DataFrame(dataset, columns=columns)
        
        return dataset_df
    
    def create_windows(self, dataset, window_size=200, stride=100, overlap=0.5):

        if stride is None:
            stride = int(window_size * (1 - overlap))
        
        feature_cols = [col for col in dataset.columns if col != 'activity']
        n_features = len(feature_cols)
        
        X_windows = []
        y_windows = []
        
        # Group by activity to maintain temporal structure within activities
        for activity in dataset['activity'].unique():
            activity_data = dataset[dataset['activity'] == activity].reset_index(drop=True)
            
            # Create windows for this activity
            for start_idx in range(0, len(activity_data) - window_size + 1, stride):
                end_idx = start_idx + window_size
                
                # Extract window
                window_data = activity_data[feature_cols].iloc[start_idx:end_idx].values
                window_label = activity_data['activity'].iloc[start_idx]
                
                # Reshape to [n_features, window_size] for CNN
                window_data = window_data.T
                
                X_windows.append(window_data)
                y_windows.append(window_label)
        
        X_windows = np.array(X_windows)
        y_windows = np.array(y_windows)
        
        return X_windows, y_windows
    
    def prepare_data(self, sensor_types=["attitude", "userAcceleration"], 
                    activities=None, window_size=200, stride=100,
                    test_size=0.2, normalize=True):
        # Create time series dataset
        dataset = self.create_time_series(sensor_types, activities, mode="raw")
        
        # Create windows
        X, y = self.create_windows(dataset, window_size, stride)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Normalize if requested
        if normalize:
            # Reshape for scaling
            n_samples, n_features, n_timesteps = X_train.shape
            X_train_flat = X_train.reshape(-1, n_features * n_timesteps)
            X_test_flat = X_test.reshape(-1, n_features * n_timesteps)
            
            # Fit scaler on train data
            X_train_flat = self.scaler.fit_transform(X_train_flat)
            X_test_flat = self.scaler.transform(X_test_flat)
            
            # Reshape back
            X_train = X_train_flat.reshape(n_samples, n_features, n_timesteps)
            X_test = X_test_flat.reshape(-1, n_features, n_timesteps)
        
        # Convert to float32 for PyTorch
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        
        data_info = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'n_classes': len(np.unique(y_encoded)),
            'n_features': X_train.shape[1],
            'window_size': window_size,
            'activity_labels': self.label_encoder.classes_,
            'sensor_types': sensor_types
        }
        
        return data_info


class MotionSenseDataset(Dataset):
    """PyTorch Dataset for MotionSense data"""
    
    def __init__(self, X, y, transform=None):

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label


def create_data_loaders(data_info, batch_size=32, num_workers=0):

    train_dataset = MotionSenseDataset(data_info['X_train'], data_info['y_train'])
    test_dataset = MotionSenseDataset(data_info['X_test'], data_info['y_test'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    loader = MotionSenseDataLoader("./data")
    
    # Prepare data with default settings
    data_info = loader.prepare_data(
        sensor_types=["attitude", "userAcceleration"],
        window_size=200,
        stride=100,
        test_size=0.2
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(data_info, batch_size=32)
    
    print("\nData loading example:")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        if batch_idx >= 2:  # Show first 3 batches
            break
