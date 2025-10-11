# 🔧 PrognosAI: AI-Driven Predictive Maintenance System Using Time-Series Sensor Data

## 🎯 Project Objective

Design and develop an AI-based predictive maintenance system to estimate the Remaining Useful Life (RUL) of industrial machinery using multivariate time-series sensor data. This prototype uses the NASA CMAPSS dataset and is adaptable across domains like turbines, pumps, and motors. The goal is to enable timely maintenance decisions, minimize unplanned downtime, and optimize asset utilization with deep learning models such as LSTM for sequential pattern recognition and failure prediction.

---

## 🗂️ Project Workflow

1. **Data Ingestion**  
   - Load and preprocess the CMAPSS sensor dataset (cycle-wise engine data).

2. **Feature Engineering**  
   - Create rolling window sequences and compute Remaining Useful Life (RUL) targets.

3. **Model Training**  
   - Train a time-series model (e.g., LSTM or GRU) to predict RUL from sensor sequences.

4. **Model Evaluation**  
   - Evaluate model performance using RMSE and compare predicted RUL vs actual RUL.

5. **Risk Thresholding**  
   - Define thresholds to trigger maintenance alerts based on predicted RUL.

6. **Visualization & Output**  
   - Present results through charts and dashboards showing RUL trends and alert zones.

---

## 🏗️ Architecture Diagram

![Architecture Diagram](path_to_your_diagram.png)  

---

## 🛠️ Tech Stack

- **Python** – Core programming language  
- **Pandas, NumPy** – Data processing  
- **Matplotlib, Seaborn** – Visualization  
- **TensorFlow / Keras** – LSTM model training  
- **Scikit-learn** – Metrics & preprocessing  
- **Streamlit / Flask** – Dashboard or API interface  
- **Docker** – Optional deployment  
- **NASA CMAPSS Dataset** – Source data  

---

## 📅 Project Milestones

### Milestone 1: Data Preparation & Feature Engineering  
- **Objective:** Load, preprocess, and prepare the CMAPSS dataset with rolling sequences and RUL targets.  
- **Deliverables:**  
  - Cleaned & preprocessed CMAPSS sensor data  
  - Python scripts for loading and preprocessing  
  - Rolling window sequences and RUL computations  
- **Evaluation:**  
  - Data integrity check  
  - Correct sequence generation  
  - Accurate RUL targets  
  - Documented data preparation  

---

### Milestone 2: Model Development & Training  
- **Objective:** Develop and train an LSTM/GRU deep learning model for RUL prediction.  
- **Deliverables:**  
  - Model architecture implementation  
  - Trained model weights  
  - Loss curves (training & validation)  
  - Code for model training and saving  
- **Evaluation:**  
  - Training convergence  
  - Validation performance inspection  
  - Model implementation correctness  

---

### Milestone 3: Model Evaluation & Performance Assessment  
- **Objective:** Evaluate model accuracy and analyze predictive performance.  
- **Deliverables:**  
  - RMSE scores on test set  
  - Plots of predicted vs actual RUL  
  - Bias and error analysis  
  - Detailed evaluation report  
- **Evaluation:**  
  - RMSE within acceptable range  
  - Visual consistency of plots  
  - Understanding model limits  

---

### Milestone 4: Risk Thresholding & Alert System  
- **Objective:** Translate RUL predictions into actionable alerts for maintenance.  
- **Deliverables:**  
  - Defined RUL alert thresholds (warning, critical)  
  - Logic for triggering alerts  
  - Sample alert instances  
- **Evaluation:**  
  - Early failure detection effectiveness  
  - Alert clarity and accuracy  
  - Practical application of alerts  

---

### Milestone 5: Visualization & Dashboard Development  
- **Objective:** Build interactive visuals and dashboards for RUL insights and alerts.  
- **Deliverables:**  
  - Interactive charts of RUL trends  
  - Dashboard with prediction and alert zones  
  - User-friendly interface (Streamlit/Flask)  
- **Evaluation:**  
  - Clarity and informativeness  
  - Responsiveness and usability  
  - Effective communication of asset health  

---

## 🚀 Quick Start: prognosAI Project

Follow these simple steps to set up and test the project:

---

> ### 🛠️ **Clone the repository**  
> ```
> git clone git@github.com:prognosAI-Infosys-Intern-project/prognosAI-Intern-project.git
> ```

---

> ### 📁 **Navigate to the project folder**  
> ```
> cd prognosAI-Intern-project/Project
> ```

---

> ### ✨ **Run Data Preprocessing**  
> Generate the required datasets:  
> ```
> python data_preprocessing.py
> ```

---

> ### 🖥️ **Start the Streamlit App**  
> Launch the web interface:  
> ```
> streamlit run app.py
> ```

---

> ### 📤 **Test Model Output**  
> Upload the `sequence` and `metadata` files from:  
> ```
> processed_data/test
> ```  
> View the results in the app.

---