# anomaly-detection-cpu
Streamlit app for real-time anomaly detection in CPU utilization data using a custom neural network
# Anomaly Detection in CPU Utilization

## Overview
This web application utilizes a Streamlit interface and a custom-built neural network model to detect anomalies in CPU utilization data. The application processes uploaded CSV files containing time-series data of CPU usage, applies anomaly detection techniques, and visualizes the results.

## Features
- Upload and process CSV data containing CPU utilization over time.
- Train a neural network model on the uploaded data using federated learning concepts.
- Visualize CPU utilization, reconstruction errors, and detected anomalies.
- Interactive Streamlit web interface for easy use and navigation.

## Installation
To run this application locally, you need to set up a Python environment and install the necessary dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-github-username/anomaly-detection-cpu.git
   ```

2. Navigate to the project directory:
   ```bash
   cd anomaly-detection-cpu
   ```

3. Create a virtual environment:
   ```bash
   python -m venv myenv
   ```

4. Activate the virtual environment:
   - Windows:
     ```bash
     myenv\\Scripts\\activate
     ```
   - macOS and Linux:
     ```bash
     source myenv/bin/activate
     ```

5. Install the required packages:
   ```bash
   pip install streamlit pandas torch matplotlib numpy
   ```

6. Run the application:
   ```bash
   streamlit run web_app.py
   ```

## Usage
- Start the web application using the `streamlit run` command.
- Navigate to the provided local URL (usually `http://localhost:8501`).
- Upload a CSV file with timestamp and CPU utilization data.
- View the processed data, anomalies detected, and corresponding visualizations.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.


