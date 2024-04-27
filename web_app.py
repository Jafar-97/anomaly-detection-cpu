# pip install virtualenv
# python -m venv myenv
# myenv\Scripts\activate
# pip install streamlit pandas torch matplotlib numpy
# streamlit run web_app.py

import streamlit as st  # Importing Streamlit for creating web apps
import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
import torch  # Importing PyTorch for deep learning operations
from torch import nn, optim  # Importing neural network and optimizer from PyTorch
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Defining a neural network model for anomaly detection
class AnomalyDetectionModel(nn.Module):
    def __init__(self):
        super(AnomalyDetectionModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 1)  # Second fully connected layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Applying ReLU activation function after the first layer
        x = self.fc2(x)  # Output of the second layer
        return x

# Function to preprocess the uploaded data
def preprocess_data(uploaded_file):
    data = pd.read_csv(uploaded_file)  # Reading data from the uploaded file
    data['timestamp'] = pd.to_datetime(data['timestamp'])  # Converting the timestamp column to datetime objects
    data['value'] = (data['value'] - data['value'].mean()) / data['value'].std()  # Standardizing the value column
    data_tensor = torch.tensor(data['value'].values, dtype=torch.float32).view(-1, 1)  # Converting data to a PyTorch tensor
    return data, data_tensor

# Function to train the model using Federated Learning
def train_model(data_tensor):
    epochs = 5  # Number of epochs for training
    batch_size = 128  # Batch size for training
    global_model = AnomalyDetectionModel()  # Initializing the global model

    for epoch in range(epochs):
        global_updates = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}  # Dictionary to store parameter updates
        total_batches = 0

        for batch_start in range(0, len(data_tensor), batch_size):
            batch_data = data_tensor[batch_start:batch_start + batch_size]  # Slicing data into batches
            local_model = AnomalyDetectionModel()  # Creating a new local model for each batch
            local_optimizer = optim.SGD(local_model.parameters(), lr=0.01)  # Initializing the optimizer
            criterion = nn.MSELoss()  # Loss function

            for _ in range(5):  # Local training loop
                local_optimizer.zero_grad()
                prediction = local_model(batch_data)
                loss = criterion(prediction, batch_data)
                loss.backward()
                local_optimizer.step()

            with torch.no_grad():
                for (name, global_param), local_param in zip(global_model.named_parameters(), local_model.parameters()):
                    global_updates[name] += local_param.clone().detach() - global_param  # Aggregating parameter updates

            total_batches += 1

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param += global_updates[name] / total_batches  # Applying the averaged updates to the global model

    return global_model

# Function to detect anomalies using the trained model
def detect_anomalies(global_model, data_tensor, data):
    global_model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        predictions = global_model(data_tensor)  # Making predictions
        reconstruction_errors = torch.abs(data_tensor - predictions)  # Calculating absolute errors
        threshold = np.percentile(reconstruction_errors.numpy(), 95)  # Determining the 95th percentile as the threshold
        anomalies = reconstruction_errors > threshold  # Identifying points where the error exceeds the threshold
    return predictions, reconstruction_errors, threshold, anomalies

# Streamlit user interface code
st.title('Anomaly Detection in CPU Utilization Data')  # Title of the web app

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])  # File uploader widget
if uploaded_file is not None:
    data, data_tensor = preprocess_data(uploaded_file)  # Preprocess the uploaded file
    global_model = train_model(data_tensor)  # Train the model
    predictions, errors, threshold, anomalies = detect_anomalies(global_model, data_tensor, data)  # Detect anomalies

    st.subheader('CPU Utilization Over Time')  # Subheader for plots
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))  # Creating a figure with 3 subplots
    ax[0].plot(data['timestamp'], data['value'], label='CPU Utilization')  # Plotting CPU utilization
    ax[1].plot(data['timestamp'], errors.numpy(), label='Reconstruction Errors')  # Plotting reconstruction errors
    ax[1].axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')  # Drawing the threshold line
    ax[2].plot(data['timestamp'], data['value'], label='CPU Utilization')  # Plotting CPU utilization again
    ax[2].scatter(data['timestamp'][anomalies.numpy().flatten()], data['value'][anomalies.numpy().flatten()], color='red', label='Anomalies')  # Marking anomalies
    
    for a in ax:
        a.legend()  # Adding legends to plots
        a.grid(True)  # Adding grid for better visibility
    st.pyplot(fig)  # Displaying the figure in the Streamlit app




