import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("merged_data/merged_all_years.csv")
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df.set_index('Date', inplace=True)

df.sort_index(inplace=True)

# Assuming data is a DataFrame and you want to create sequences from it
def create_sequences(data, time_steps=24):
    sequences = []
    labels = []
    
    # Ensure the data is in the correct shape for time series
    for i in range(len(data) - 60):
        # Input sequence: 24 time steps (1 day of data)
        sequence = data.iloc[i:i+time_steps].values  # This will be a (24, num_features) array
        
        # Output: The next day's data (24 time steps)
        label = data.iloc[i+time_steps:i+2*time_steps].values  # Next day's values (24 time steps)
        
        sequences.append(sequence)
        labels.append(label)
    
    # Convert to NumPy arrays (should have shape: (num_samples, time_steps, num_features))
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    return sequences, labels

scaler = MinMaxScaler(feature_range=(0, 1))
label_scaler = MinMaxScaler(feature_range=(0, 1))

features = df.loc[:, ['Temperature', 'Relative Humidity', 'WEST']]

label_scaler.fit(features[['WEST']])
features[['Temperature', 'Relative Humidity']] = scaler.fit_transform(features[['Temperature', 'Relative Humidity']])
features[['WEST']] = label_scaler.transform(features[['WEST']])

sequences, labels = create_sequences(features, time_steps=24)

# Normalize the data (using MinMaxScaler)

num_features = sequences.shape[2]

print(f"Shape of sequences: {sequences.shape}")
labels = labels[:,:,2]
print(f"Shape of labels: {labels.shape}")

train_size = int(len(sequences) * 0.8)
X_train, y_train = sequences[:train_size], labels[:train_size]
X_test, y_test = sequences[train_size:], labels[train_size:]

import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])  # Get the last time step's output
        return predictions
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
input_size = num_features
hidden_size = 64
output_size = 24

model = LSTMModel(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32

train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
minloss = float('inf')
losses = []
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred.squeeze(), batch_y)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        losses.append(loss_value)
        if loss_value < minloss:
            minloss = loss_value
            torch.save(model.state_dict(), 'best_model.pth')

model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
# Example to make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    future_input = torch.Tensor(X_test).to(device)  # Prepare your test data
    future_predictions = model(future_input).cpu()

# Convert predictions back to the original scale
predicted_load = label_scaler.inverse_transform(future_predictions.numpy().reshape(-1, 24))
actual_load = label_scaler.inverse_transform(y_test)

# Visualize the predictions
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

axs[0].plot(predicted_load[0], label='Predicted Load')
axs[0].plot(actual_load[0], label='Actual Load', linestyle='--')
axs[0].set_title('Predicted vs Actual Load for the Next Day')
axs[0].set_xlabel('Time Steps')
axs[0].set_ylabel('Load (WEST)')
axs[0].legend()

axs[1].plot(predicted_load[287], label='Predicted Load')
axs[1].plot(actual_load[287], label='Actual Load', linestyle='--')
axs[1].set_title('Predicted vs Actual Load for the Next Day')
axs[1].set_xlabel('Time Steps')
axs[1].set_ylabel('Load (WEST)')
axs[1].legend()

axs[2].plot(predicted_load[90], label='Predicted Load')
axs[2].plot(actual_load[90], label='Actual Load', linestyle='--')
axs[2].set_title('Predicted vs Actual Load for the Next Day')
axs[2].set_xlabel('Time Steps')
axs[2].set_ylabel('Load (WEST)')
axs[2].legend()

plt.tight_layout()
plt.savefig('predicted_load.png')

plt.clf()
plt.plot(losses)
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('losses.png')