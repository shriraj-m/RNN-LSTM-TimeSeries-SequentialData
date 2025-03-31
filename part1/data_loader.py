import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class HARDataset(Dataset):
    def __init__(self, features, labels, sequence_length=10):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length
        
        # Calculate the number of complete sequences
        self.num_sequences = len(self.labels) // sequence_length
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Get the start index for this sequence
        start_idx = idx * self.sequence_length
        
        # Get sequence of features and labels
        sequence = self.features[start_idx:start_idx + self.sequence_length]
        label = self.labels[start_idx + self.sequence_length - 1]
        
        return sequence, label

def aggregate_features(features, window_size=5):
    """
    Aggregate features using sliding window statistics
    """
    n_samples = len(features)
    n_features = features.shape[1]
    aggregated_features = []
    
    for i in range(0, n_samples - window_size + 1):
        window = features[i:i + window_size]
        # Calculate statistics for each feature
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        max_val = np.max(window, axis=0)
        min_val = np.min(window, axis=0)
        
        # Concatenate statistics
        aggregated = np.concatenate([mean, std, max_val, min_val])
        aggregated_features.append(aggregated)
    
    return np.array(aggregated_features)

def load_data(data_path, n_splits=5):
    """
    load the HAR dataset from the specified path and prepare for k-fold cross-validation
    """
    # load training data
    train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    
    # create label encoder for activities
    activity_labels = {
        'WALKING': 0,
        'WALKING_UPSTAIRS': 1,
        'WALKING_DOWNSTAIRS': 2,
        'SITTING': 3,
        'STANDING': 4,
        'LAYING': 5
    }
    
    # convert activity labels to numerical values
    train_data['Activity'] = train_data['Activity'].map(activity_labels)
    test_data['Activity'] = test_data['Activity'].map(activity_labels)
    
    # separate features and labels
    X_train = train_data.drop('Activity', axis=1).values
    y_train = train_data['Activity'].values
    X_test = test_data.drop('Activity', axis=1).values
    y_test = test_data['Activity'].values
    
    # scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Aggregate features
    X_train_agg = aggregate_features(X_train)
    X_test_agg = aggregate_features(X_test)
    
    # Create k-fold cross-validation splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = []
    
    for train_idx, val_idx in kf.split(X_train_agg):
        cv_splits.append({
            'train': (X_train_agg[train_idx], y_train[train_idx]),
            'val': (X_train_agg[val_idx], y_train[val_idx])
        })
    
    return cv_splits, (X_test_agg, y_test)

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, sequence_length=10):
    """
    create PyTorch DataLoaders for training, validation and testing
    """
    # create datasets
    train_dataset = HARDataset(X_train, y_train, sequence_length)
    val_dataset = HARDataset(X_val, y_val, sequence_length)
    test_dataset = HARDataset(X_test, y_test, sequence_length)
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

def get_data_loaders(data_path, batch_size=32, sequence_length=10, n_splits=5):
    """
    main function to load and prepare data with k-fold cross-validation
    """
    # load and preprocess data
    cv_splits, (X_test, y_test) = load_data(data_path, n_splits)
    
    # Create data loaders for each fold
    fold_loaders = []
    for fold in cv_splits:
        X_train, y_train = fold['train']
        X_val, y_val = fold['val']
        
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size, sequence_length
        )
        fold_loaders.append({
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        })
    
    return fold_loaders
