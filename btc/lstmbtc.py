import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

# Load the CSV data
csv_filename = "bitcoin_price_inr.csv"
df = pd.read_csv(csv_filename)

# Extract the price column and convert it to numpy array
prices = df["Price"].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Define a function to create sequences of data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define hyperparameters
sequence_length = 10  # Number of time steps to look back
train_size = int(len(prices_scaled) * 0.67)
test_size = len(prices_scaled) - train_size

# Split the data into training and testing sets
train_data = prices_scaled[0:train_size, :]
test_data = prices_scaled[train_size:len(prices_scaled), :]

# Create sequences for training and testing
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(1, 64, 2, 1).to(device)

# Define loss and optimization functions
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train.to(device))
    optimizer.zero_grad()
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
X_test = X_test.to(device)
y_pred = model(X_test)
y_pred = scaler.inverse_transform(y_pred.detach().cpu().numpy())
y_true = scaler.inverse_transform(y_test.detach().cpu().numpy())

# Calculate RMSE (Root Mean Squared Error) to evaluate the model
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.title("Bitcoin Price Prediction with LSTM")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.plot(df.index[train_size + sequence_length:], y_true, label="True Price", color="green")
plt.plot(df.index[train_size + sequence_length:], y_pred, label="Predicted Price", color="blue")
plt.legend()
plt.show()

