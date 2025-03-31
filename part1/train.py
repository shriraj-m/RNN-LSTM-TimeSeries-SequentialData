import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import kagglehub
from datetime import datetime

from data_loader import get_data_loaders
from model import get_model

class ModelTrainer:
    def __init__(self, model_type, input_size, hidden_size, num_classes, num_layers, device):
        self.model_type = model_type
        self.device = device
        self.model = get_model(model_type, input_size, hidden_size, num_classes, num_layers)
        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
    def train_model(self, train_loader, val_loader, test_loader, num_epochs):
        print(f'\nTraining {self.model_type.upper()} model...')
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            val_loss, val_acc, _, _ = self.evaluate_model(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Load best model for final test evaluation
        self.model.load_state_dict(best_model_state)
        test_loss, test_acc, test_predictions, test_labels = self.evaluate_model(test_loader)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        
        print(f'\nFinal Test Results for {self.model_type.upper()}:')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Plot confusion matrix after all training is complete
        self.plot_confusion_matrix(test_predictions, test_labels)
    
    def evaluate_model(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        # print classification report
        # print(f'\nClassification Report for {self.model_type.upper()}:')
        # print(classification_report(all_labels, all_predictions))
        
        return avg_loss, accuracy, all_predictions, all_labels

    def plot_confusion_matrix(self, predictions, labels):
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'aggregated_confusion_matrix_{self.model_type}.png')
        plt.close()

def plot_comparison(trainers):
    # plot training and validation losses
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    for trainer in trainers:
        plt.plot(trainer.train_losses, label=f'{trainer.model_type.upper()} Train')
        plt.plot(trainer.val_losses, '--', label=f'{trainer.model_type.upper()} Val')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # plot training and validation accuracies
    plt.subplot(1, 3, 2)
    for trainer in trainers:
        plt.plot(trainer.train_accuracies, label=f'{trainer.model_type.upper()} Train')
        plt.plot(trainer.val_accuracies, '--', label=f'{trainer.model_type.upper()} Val')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # plot testing accuracies
    plt.subplot(1, 3, 3)
    for trainer in trainers:
        plt.plot(trainer.test_accuracies, label=f'{trainer.model_type.upper()} Test')
    plt.title('Testing Accuracies')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('aggregated_model_comparison.png')
    plt.close()
    
    # print final results
    print('\nFinal Results:')
    print('Model\t\tFinal Train Acc\tFinal Val Acc\tFinal Test Acc')
    print('-' * 70)
    for trainer in trainers:
        print(f'{trainer.model_type.upper():<12} {trainer.train_accuracies[-1]:.2f}%\t\t{trainer.val_accuracies[-1]:.2f}%\t\t{trainer.test_accuracies[-1]:.2f}%')

def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # hyperparameters
    input_size = 2248  # 562 features * 4 (mean, std, max, min) for aggregated features\
    #input_size = 562  # number of features in the HAR dataset
    hidden_size = 64
    num_classes = 6  # number of activity classes
    num_layers = 1
    batch_size = 32
    sequence_length = 10
    num_epochs = 70
    n_splits = 5 # number of folds for cross-validation
    
    # download dataset
    data_path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
    print("Path to dataset files:", data_path)
    
    # get data loaders with k-fold cross-validation
    fold_loaders = get_data_loaders(data_path, batch_size, sequence_length, n_splits)
    
    # list of models to train
    model_types = ['rnn', 'lstm', 'bilstm'] # can throw in 'gru' as well
    trainers = []
    
    # train each model on each fold
    for model_type in model_types:
        fold_trainer = ModelTrainer(model_type, input_size, hidden_size, num_classes, num_layers, device)
        
        # Train on each fold
        for fold_idx, fold_data in enumerate(fold_loaders):
            print(f'\nTraining {model_type.upper()} on Fold {fold_idx + 1}/{n_splits}')
            fold_trainer.train_model(
                fold_data['train'],
                fold_data['val'],
                fold_data['test'],
                num_epochs
            )
        
        trainers.append(fold_trainer)
    
    # plot comparison
    plot_comparison(trainers)

if __name__ == '__main__':
    main()  