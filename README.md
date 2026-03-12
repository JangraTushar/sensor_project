<h1 align="center">🔬 Wafer Sensor Fault Detection</h1>

<p align="center">
  <b>An end-to-end Machine Learning system to detect faulty semiconductor wafer sensors using MongoDB, Scikit-learn, XGBoost, and Flask.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/MongoDB-Database-green?style=for-the-badge&logo=mongodb" />
  <img src="https://img.shields.io/badge/XGBoost-Model-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge&logo=scikit-learn" />
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Core Modules — Detailed Breakdown](#-core-modules--detailed-breakdown)
- [ML Pipeline Flow](#-ml-pipeline-flow)
- [Installation & Setup](#-installation--setup)
- [Environment Variables](#-environment-variables)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Model Selection Strategy](#-model-selection-strategy)
- [Configuration Files](#-configuration-files)
- [Logging & Exception Handling](#-logging--exception-handling)
- [Artifacts & Output](#-artifacts--output)
- [Future Improvements](#-future-improvements)

---

## 🧠 Overview

The **Wafer Sensor Fault Detection** system is a production-ready binary classification pipeline built for the semiconductor manufacturing industry. It ingests multi-sensor data from a MongoDB database, applies automated preprocessing, trains and fine-tunes multiple ML models, selects the best performer, and serves predictions through a Flask web application — all through clean, modular, industry-standard code.

---

## 🎯 Problem Statement

In semiconductor wafer fabrication, thousands of sensors monitor each step of the process. A faulty sensor can lead to defective wafers and significant financial loss. This system automates the detection of whether a sensor is **Good** or **Bad** by:

- Ingesting raw multi-sensor data stored in MongoDB
- Preprocessing and scaling the features
- Training and comparing multiple classifiers with hyperparameter tuning
- Serving predictions via CSV upload through a web interface

The target column `quality` is mapped as:
- `-1` → `0` (Bad)
- `1` → `1` (Good)

---

## 🏗 System Architecture

```
MongoDB (Raw Data)
        │
        ▼
 [ Data Ingestion ]          ← Pulls data from MongoDB, saves as CSV
        │
        ▼
 [ Data Transformation ]     ← Imputation → RobustScaling → Train/Test Split
        │
        ▼
 [ Model Trainer ]           ← Evaluate 4 models → Fine-tune best → Save .pkl
        │
        ▼
 [ Training Pipeline ]       ← Orchestrates all 3 components end-to-end
        │
        ▼
 [ Prediction Pipeline ]     ← Upload CSV → Preprocess → Predict → Download CSV
        │
        ▼
   Flask Web App             ← /train and /predict routes exposed
```

---

## 🛠 Tech Stack

| Category         | Technology                                          |
|------------------|-----------------------------------------------------|
| Language         | Python 3.8+                                         |
| Web Framework    | Flask                                               |
| Database         | MongoDB Atlas (via PyMongo)                         |
| ML Libraries     | Scikit-learn, XGBoost                               |
| Data Processing  | Pandas, NumPy                                       |
| Configuration    | YAML, Python dataclasses                            |
| Serialization    | Pickle                                              |
| Logging          | Python `logging` module (timestamped log files)     |
| Cloud (optional) | AWS S3 via Boto3                                    |
| Frontend         | HTML5, CSS3 (file upload form)                      |

---

## 📁 Project Structure

```
sensor_project/
│
├── app.py                          # Flask application entry point
│
├── src/                            # Core source package
│   ├── __init__.py
│   ├── exception.py                # Custom exception with file/line tracing
│   ├── logger.py                   # Timestamped rotating log file setup
│   │
│   ├── constant/
│   │   └── __init__.py             # Global constants: MongoDB URL, DB/collection names, artifact paths
│   │
│   ├── components/                 # ML pipeline components (Data → Train)
│   │   ├── data_ingestion.py       # Fetch data from MongoDB → save as CSV
│   │   ├── data_transformation.py  # Impute → Scale → Split → Save preprocessor.pkl
│   │   └── model_trainer.py        # Train → Evaluate → Fine-tune → Save model.pkl
│   │
│   ├── pipeline/                   # Orchestration pipelines
│   │   ├── train_pipeline.py       # End-to-end training pipeline
│   │   └── predict_pipeline.py     # CSV upload → prediction → downloadable output
│   │
│   └── utils/
│       └── main_utils.py           # YAML reader, pickle save/load, schema reader
│
├── config/
│   ├── model.yaml                  # GridSearchCV hyperparameter grids for all models
│   └── schema.yaml                 # Input data schema definition
│
├── artifacts/                      # Auto-generated: model.pkl, preprocessor.pkl, wafer_fault.csv
│
├── prediction_artifacts/           # Uploaded CSV files for prediction (auto-created)
│
├── predictions/                    # Output: prediction_file.csv (auto-created)
│
├── notebooks/                      # EDA and experimentation notebooks
│
├── templates/
│   └── upload_file.html            # Drag-and-drop CSV upload UI
│
├── static/
│   └── css/
│       └── style.css               # Frontend styling
│
├── .gitignore
└── README.md
```

---

## 🔍 Core Modules — Detailed Breakdown

### 1. `src/constant/__init__.py` — Global Configuration
Defines all project-wide constants used across every module:

| Constant                 | Value                   | Purpose                            |
|--------------------------|-------------------------|------------------------------------|
| `MONGO_DATABASE_NAME`    | `"pwskills"`            | MongoDB database name              |
| `MONGO_COLLECTION_NAME`  | `"waferfault"`          | MongoDB collection name            |
| `MONGO_DB_URL`           | Connection string       | Atlas connection URI               |
| `TARGET_COLUMN`          | `"quality"`             | ML target label column             |
| `artifact_folder`        | `"artifacts"`           | Root path for all saved artifacts  |
| `MODEL_FILE_NAME`        | `"model"`               | Base name for the saved model file |
| `MODEL_FILE_EXTENSION`   | `".pkl"`                | Serialization format               |

---

### 2. `src/exception.py` — Custom Exception Handler
Wraps Python's base `Exception` with rich diagnostic output:
- Captures **filename**, **line number**, and **error message** using `sys.exc_info()`
- Raises a `CustomException` with a formatted, human-readable message
- Used consistently across **every** `try/except` block in the project

```python
# Example output:
# "Error occurred in python script name [data_ingestion.py] line number  error message [...]"
```

---

### 3. `src/logger.py` — Timestamped File Logger
- Creates a new log file on every run named after the current timestamp: `MM_DD_YYYY_HH_MM_SS.log`
- Stores logs under a `logs/` directory (auto-created)
- Format: `[timestamp] line_number module_name - LEVEL - message`
- Level: `INFO` by default

---

### 4. `src/utils/main_utils.py` — Shared Utilities (`MainUtils` class)

| Method                  | Description                                          |
|-------------------------|------------------------------------------------------|
| `read_yaml_file()`      | Reads any YAML config file and returns a dict        |
| `read_schema_config_file()` | Reads `config/schema.yaml` specifically         |
| `save_object()`         | Serializes and saves any Python object using pickle  |
| `load_object()`         | Loads and deserializes a pickled object from disk    |

Also imports `boto3` for optional AWS S3 integration.

---

### 5. `src/components/data_ingestion.py` — Data Ingestion Component

**Config class (`DataIngestionConfig`):**
- `artifact_folder` → path to store `wafer_fault.csv`

**Key methods:**

| Method                                  | What it does                                                     |
|-----------------------------------------|------------------------------------------------------------------|
| `export_collection_as_dataframe()`      | Connects to MongoDB → fetches collection → returns a DataFrame   |
| `export_data_into_feature_store_file_path()` | Calls above, removes `_id` col, replaces `"na"` with NaN, saves CSV |
| `initiate_data_ingestion()`             | Entry point → returns path of the saved CSV file                 |

---

### 6. `src/components/data_transformation.py` — Data Transformation Component

**Config class (`DataTransformationConfig`):**
- `transformed_train_file_path` → `artifacts/train.npy`
- `transformed_test_file_path` → `artifacts/test.npy`
- `transformed_object_file_path` → `artifacts/preprocessor.pkl`

**Key methods:**

| Method                           | What it does                                                             |
|----------------------------------|--------------------------------------------------------------------------|
| `get_data()`                     | Reads CSV, renames `"Good/Bad"` column to `TARGET_COLUMN`                |
| `get_data_transformer_object()`  | Builds an sklearn Pipeline: `SimpleImputer(constant=0)` + `RobustScaler` |
| `initiate_data_transformation()` | Splits data (80/20), fits preprocessor, saves `.pkl`, returns numpy arrays|

**Target encoding:** Maps `-1` → `0` (Bad) and `1` → `1` (Good) using `np.where`.

---

### 7. `src/components/model_trainer.py` — Model Trainer Component

**Config class (`ModelTrainerConfig`):**
- `trained_model_path` → `artifacts/model.pkl`
- `expected_accuracy` → `0.45` (minimum threshold)
- `model_config_file_path` → `config/model.yaml`

**Models evaluated:**

| Model                       | Library         |
|-----------------------------|-----------------|
| `XGBClassifier`             | XGBoost         |
| `GradientBoostingClassifier`| Scikit-learn    |
| `SVC`                       | Scikit-learn    |
| `RandomForestClassifier`    | Scikit-learn    |

**Key methods:**

| Method                 | What it does                                                              |
|------------------------|---------------------------------------------------------------------------|
| `evaluate_models()`    | Trains all models, computes accuracy on test set, returns a score dict    |
| `get_best_model()`     | Picks the model with the highest test accuracy                            |
| `finetune_best_model()`| Runs `GridSearchCV` (cv=5, n_jobs=-1) using params from `model.yaml`      |
| `initiate_model_trainer()` | Full flow: evaluate → select best → fine-tune → fit → save → return path |

Raises an exception if no model exceeds the accuracy threshold.

---

### 8. `src/pipeline/train_pipeline.py` — Training Pipeline Orchestrator

`TrainingPipeline` class ties all three components together sequentially:

```
run_pipeline()
    ├── start_data_ingestion()       → returns feature_store_file_path
    ├── start_data_transformation()  → returns train_arr, test_arr, preprocessor_path
    └── start_model_training()       → returns trained model path / score
```

Triggered via the `/train` Flask route.

---

### 9. `src/pipeline/predict_pipeline.py` — Prediction Pipeline

**Config class (`PredictionPipelineConfig`):**
- `model_file_path` → `artifacts/model.pkl`
- `preprocessor_path` → `artifacts/preprocessor.pkl`
- `prediction_file_path` → `predictions/prediction_file.csv`

**Key methods:**

| Method                       | What it does                                                          |
|------------------------------|-----------------------------------------------------------------------|
| `save_input_files()`         | Reads uploaded CSV from Flask `request`, saves to `prediction_artifacts/` |
| `predict()`                  | Loads model + preprocessor, transforms input, returns predictions     |
| `get_predicted_dataframe()`  | Appends predictions as `quality` column mapped to `"good"` / `"bad"` |
| `run_pipeline()`             | Orchestrates save → predict → write CSV → return config               |

Output CSV is directly streamed back to the user as a download.

---

### 10. `app.py` — Flask Web Application Entry Point

| Route       | Method      | Description                                                              |
|-------------|-------------|--------------------------------------------------------------------------|
| `/`         | GET         | Health check — returns `"Welcome to my application"`                     |
| `/train`    | GET         | Triggers the full `TrainingPipeline.run_pipeline()` end-to-end           |
| `/predict`  | GET         | Renders the CSV upload HTML form (`upload_file.html`)                    |
| `/predict`  | POST        | Runs `PredictionPipeline`, returns downloadable `prediction_file.csv`    |

Runs on `host="0.0.0.0"`, `port=5000` in debug mode.

---

### 11. `config/model.yaml` — Model Hyperparameter Grid

Defines the `GridSearchCV` search space for each model:

```yaml
model_selection:
  model:
    XGBClassifier:
      search_param_grid:
        learning_rate: [0.1, 0.01, 0.001]
        max_depth:[1][2][3]
        n_estimators: 
        gamma: [0, 0.1, 0.2]

    GradientBoostingClassifier:
      search_param_grid:
        n_estimators: 
        criterion: [friedman_mse]
        # ... more params
```

---

### 12. `templates/upload_file.html` — Prediction UI

A clean drag-and-drop HTML interface:
- Accepts `.csv` file upload only
- Submits via POST to `/predict`
- On success, triggers automatic download of the prediction CSV

---

## 🔄 ML Pipeline Flow

```
MongoDB Raw Data
      │
      ▼
DataIngestion.initiate_data_ingestion()
      │  ➜ artifacts/wafer_fault.csv
      ▼
DataTransformation.initiate_data_
