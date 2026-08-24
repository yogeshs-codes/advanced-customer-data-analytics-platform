# Advanced Customer Data Analytics Platform

## Overview

This project focuses on analyzing customer purchasing behavior and building a machine learning solution to understand and predict customer demand.

The project follows an end-to-end machine learning workflow covering problem definition, data understanding, preprocessing, exploratory data analysis, feature engineering, model development, model evaluation, experiment tracking, model selection, model serving, production monitoring, and dashboard visualization.

The main objective is to use customer transaction data to identify meaningful purchasing patterns and develop a data-driven solution that can support customer-product purchase prediction and demand-related business decision-making.

## Objectives

- Understand customer purchasing and ordering behavior.
- Explore patterns and trends in customer demand.
- Perform data cleaning and preprocessing.
- Conduct exploratory data analysis (EDA).
- Engineer meaningful customer-, product-, and order-level features.
- Develop and compare suitable machine learning models.
- Tune model hyperparameters and validate model performance.
- Select a final model based on evaluation results.
- Track machine learning experiments and results.
- Serve the selected model through a REST API.
- Monitor production prediction activity, latency, probability, and model health.
- Provide a React-based model monitoring dashboard.
- Containerize the model-serving application using Docker.
- Present results and technical work through clear documentation.

## Project Workflow

The project is organized into the following major stages:

1. **Problem Definition**
   - Understand the business problem.
   - Define the machine learning objective.
   - Identify the prediction target and relevant variables.

2. **Data Collection & Understanding**
   - Obtain the required dataset.
   - Understand available tables, columns, data types, and relationships.
   - Analyze the size and structure of the data.

3. **Data Preprocessing**
   - Handle missing values.
   - Remove or address duplicate and inconsistent records.
   - Convert data into suitable formats.
   - Prepare the data for analysis and modeling.

4. **Exploratory Data Analysis**
   - Analyze customer and product behavior.
   - Identify demand patterns and trends.
   - Study distributions and relationships between variables.
   - Create visualizations to communicate important findings.

5. **Feature Engineering**
   - Create customer-product behavioral features.
   - Create customer-department and customer-aisle features.
   - Create product-level purchase and reorder features.
   - Prepare the final feature set for machine learning.

6. **Model Development**
   - Select appropriate machine learning algorithms.
   - Train candidate models using the engineered features.
   - Compare model architectures and their performance.

7. **Hyperparameter Tuning & Validation**
   - Tune relevant model hyperparameters.
   - Evaluate models using validation data and appropriate metrics.
   - Compare tuned configurations and identify strong candidates.

8. **Experiment Tracking**
   - Track model experiments and evaluation results.
   - Record model configurations, parameters, and metrics.
   - Maintain reproducibility of the model development process.

9. **Final Model Selection & Evaluation**
   - Compare candidate models using consistent evaluation criteria.
   - Select the final model based on validation and evaluation results.
   - Document the final model and its performance.

10. **Model Serving**
    - Load the selected trained model.
    - Expose prediction functionality through a FastAPI REST API.
    - Provide health-check and prediction endpoints.
    - Validate prediction requests using Pydantic.
    - Return consistent JSON responses.

11. **Model Monitoring**
    - Record production prediction events.
    - Track prediction distribution and purchase probabilities.
    - Monitor inference latency.
    - Detect latency anomalies and prediction-distribution drift.
    - Expose monitoring metrics through the `/monitoring` endpoint.

12. **Monitoring Dashboard**
    - Provide a React/Vite operational dashboard.
    - Display model and API health.
    - Visualize prediction distribution and latency statistics.
    - Display monitoring alerts and recent prediction records.

13. **Containerization**
    - Package the model-serving application with Docker.
    - Install application dependencies inside the container.
    - Include the trained model required for inference.
    - Run the API through Uvicorn on port 8000.

14. **Insights & Conclusion**
    - Summarize key findings.
    - Identify important demand and purchasing patterns.
    - Discuss practical implications.
    - Document limitations and possible future improvements.

## Project Tasks and Deliverables

The repository contains work from the different stages of the project.

### Phase 1 - Problem Definition and Project Setup

The initial phase establishes the project objective, machine learning problem, repository structure, and overall project direction.

### Phase 2 - Machine Learning Development

The machine learning development work includes:

- Data understanding and preparation.
- Feature engineering.
- Exploratory analysis.
- Model architecture comparison.
- Hyperparameter tuning.
- Experiment tracking.
- Final model selection and validation.
- Final model evaluation and documentation.

Key documentation and implementation files are maintained under `docs/` and `src/`.

### Phase 3 - FastAPI, Docker, Model Monitoring, and Dashboard

## Task 1 - Model Serving API

The model-serving task exposes the final customer-product purchase prediction model through a FastAPI application.

The implementation includes:

- **FastAPI** REST API.
- `GET /health` endpoint for API and model readiness.
- `POST /predict` endpoint for customer-product purchase prediction.
- Pydantic request validation.
- Consistent JSON prediction responses.
- Model feature-count verification.
- Probability output for the future-purchase class.
- Swagger UI documentation.
- Docker containerization.

The serving application is implemented in:

```text
src/serving_api.py
```

The model-serving documentation is available in:

```text
docs/task1_model_serving_api.md
```

The Docker configuration is provided through:

```text
Dockerfile
.dockerignore
requirements.txt
```

The API serves the selected Gradient Boosting Classifier model using:

```text
output/models/final_selected_model.joblib
```

The serving layer verifies the expected 19 model features before performing inference.

### API Endpoints

#### `GET /health`

Checks whether the API is running and whether the trained model has been loaded successfully.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "GradientBoostingClassifier",
  "model_version": "gradient_boosting_v1",
  "feature_count": 19
}
```

#### `POST /predict`

Accepts the required model features and returns:

- Binary prediction.
- Human-readable prediction label.
- Probability of future purchase.
- Model version.
- Feature count.

The API validates required fields, numeric values, non-negative constraints, purchase/reorder rates, order hour, and day-of-week values before inference.

### Dockerized Serving

The model-serving application can be built and run from the project root using:

```bash
docker build -t customer-demand-api .
docker run --name customer-demand-container -p 8000:8000 customer-demand-api
```

After the container starts, the API is available locally at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Task 2 - Model Monitoring and Monitoring Dashboard

The model monitoring task extends the FastAPI serving layer with production-oriented monitoring of prediction activity, inference latency, prediction probabilities, prediction distribution, and model health.

The monitoring implementation records prediction events and exposes aggregated monitoring metrics through the FastAPI `/monitoring` endpoint.

### Model Monitoring Implementation

The monitoring service is implemented in:

```text
src/model_monitor.py
```

The serving API integrates the monitoring service through:

```text
src/serving_api.py
```

After a successful prediction, the serving layer records monitoring information including:

- Prediction timestamp.
- Binary prediction result.
- Probability of future purchase.
- Model version.
- Inference latency in milliseconds.

The monitoring service aggregates these records and provides operational metrics for observing model-serving behavior.

### Monitoring Metrics

The `/monitoring` endpoint provides the following metrics:

| Metric | Description |
| :--- | :--- |
| `total_predictions` | Total number of predictions processed |
| `positive_predictions` | Number of predictions classified as future purchases |
| `negative_predictions` | Number of predictions classified as no future purchase |
| `positive_prediction_rate` | Percentage of positive predictions |
| `negative_prediction_rate` | Percentage of negative predictions |
| `recent_prediction_count` | Number of recent prediction records |
| `recent_positive_prediction_rate` | Positive prediction rate in the recent prediction window |
| `average_latency_ms` | Average model inference latency |
| `minimum_latency_ms` | Minimum observed inference latency |
| `maximum_latency_ms` | Maximum observed inference latency |
| `average_probability_future_purchase` | Average predicted probability of future purchase |
| `latency_anomaly` | Indicates whether an abnormal latency condition has been detected |
| `prediction_distribution_drift` | Indicates whether prediction distribution drift has been detected |
| `alerts` | Active monitoring alerts |
| `recent_predictions` | Recent prediction-level monitoring records |

### Monitoring API Endpoint

#### `GET /monitoring`

Returns the current production monitoring status and aggregated prediction metrics.

Example response:

```json
{
  "total_predictions": 2,
  "positive_predictions": 2,
  "negative_predictions": 0,
  "positive_prediction_rate": 1.0,
  "negative_prediction_rate": 0.0,
  "recent_prediction_count": 2,
  "recent_positive_prediction_rate": 1.0,
  "average_latency_ms": 96.11,
  "minimum_latency_ms": 71.73,
  "maximum_latency_ms": 120.50,
  "average_probability_future_purchase": 0.529,
  "latency_anomaly": false,
  "prediction_distribution_drift": false,
  "alerts": []
}
```

The monitoring endpoint allows the operational state of the model-serving system to be inspected without directly accessing prediction logs.

### Monitoring and Anomaly Detection

The monitoring system checks two important operational conditions.

**Latency anomaly**

Inference latency is tracked for every prediction. Minimum, average, and maximum latency values are calculated, and abnormal latency conditions can generate monitoring alerts.

**Prediction distribution drift**

The monitoring service tracks the distribution of prediction classes and compares recent prediction behavior against the expected monitoring baseline. This helps identify unusual changes in prediction behavior.

### Recent Prediction Monitoring

The monitoring response also provides recent prediction records containing:

- Timestamp.
- Prediction class.
- Future-purchase probability.
- Model version.
- Inference latency.

Example:

```json
{
  "timestamp": "2026-08-24T14:55:22.953156+00:00",
  "prediction": 1,
  "probability_future_purchase": 0.5290054925742836,
  "model": "gradient_boosting_v1",
  "latency_ms": 71.73
}
```

### Monitoring Dashboard

A React-based monitoring dashboard is provided under:

```text
dashboard/
```

The dashboard is implemented using:

```text
dashboard/src/App.jsx
dashboard/src/App.css
dashboard/src/index.css
```

The dashboard consumes the FastAPI `/monitoring` endpoint and presents the monitoring information in an operational interface.

The dashboard provides:

- System health status.
- Model name and version.
- Algorithm information.
- Number of model features.
- API health status.
- Total prediction count.
- Positive prediction count and rate.
- Negative prediction count and rate.
- Average inference latency.
- Prediction distribution visualization.
- Latency statistics visualization.
- Latency anomaly status.
- Prediction distribution drift status.
- Average future-purchase probability.
- Recent prediction history.

The dashboard uses Recharts for monitoring visualizations.

### Dashboard Architecture

The monitoring flow is:

```text
Customer Prediction Request
          ↓
     FastAPI /predict
          ↓
     Model Inference
          ↓
   Monitoring Service
          ↓
Prediction + Probability + Latency
          ↓
     /monitoring
          ↓
React Monitoring Dashboard
```

This provides a clear separation between model inference, monitoring data collection, monitoring metrics, and dashboard visualization.

### Running the Monitoring System

Start the FastAPI serving application from the project root:

```bash
uvicorn src.serving_api:app --host 0.0.0.0 --port 8000
```

The monitoring endpoint is then available at:

```text
http://127.0.0.1:8000/monitoring
```

Start the React dashboard from the dashboard directory:

```bash
cd dashboard
npm install
npm run dev
```

The Vite development server provides the dashboard locally, normally at:

```text
http://localhost:5173
```

The dashboard communicates with the FastAPI monitoring endpoint on port `8000`.

### Monitoring Deliverables

| Component | Location |
| :--- | :--- |
| Monitoring service | `src/model_monitor.py` |
| FastAPI serving and monitoring integration | `src/serving_api.py` |
| Monitoring documentation | `docs/model_monitoring.md` |
| React dashboard | `dashboard/` |
| Dashboard application | `dashboard/src/App.jsx` |
| Dashboard styling | `dashboard/src/App.css` |
| Dashboard package configuration | `dashboard/package.json` |

## Detailed Model Serving API

The final selected customer-product purchase prediction model is exposed through a FastAPI REST API implemented in:

```text
src/serving_api.py
```

The application loads:

```text
output/models/final_selected_model.joblib
```

The serving layer verifies that the trained model contains the expected **19 features** before inference.

### API Endpoints

#### `GET /health`

Checks whether the API is running and whether the trained model has been loaded.

#### `POST /predict`

Generates a future-purchase prediction for one customer-product record.

The request is validated by Pydantic before model inference. The 19 required fields are:

| Field | Validation |
| :--- | :--- |
| `user_product_purchase_count` | >= 0 |
| `user_product_reorder_count` | >= 0 |
| `user_product_last_order_number` | >= 0 |
| `user_product_reorder_rate` | 0 to 1 |
| `user_product_avg_cart_position` | >= 0 |
| `user_product_recency_orders` | >= 0 |
| `department_id` | >= 0 |
| `user_department_purchase_count` | >= 0 |
| `user_department_purchase_share` | 0 to 1 |
| `aisle_id` | >= 0 |
| `user_aisle_purchase_count` | >= 0 |
| `user_aisle_purchase_share` | 0 to 1 |
| `user_total_orders` | >= 0 |
| `user_avg_days_between_orders` | >= 0 |
| `user_avg_order_hour` | 0 to 23 |
| `user_avg_order_dow` | 0 to 6 |
| `product_total_purchases` | >= 0 |
| `product_unique_users` | >= 0 |
| `product_reorder_rate` | 0 to 1 |

The response contains:

- `prediction` — binary prediction (`0` or `1`).
- `prediction_label` — `Future Purchase` or `No Future Purchase`.
- `probability_future_purchase` — probability of the future-purchase class.
- `model` — serving model version.
- `feature_count` — number of features used.

### Validation and Error Handling

The Pydantic request schema enforces required fields, numeric types, non-negative values, rate ranges from `0` to `1`, order hour from `0` to `23`, and day-of-week from `0` to `6`.

Invalid input is rejected with `422 Unprocessable Entity`. Unexpected prediction failures are returned as `500 Internal Server Error`.

### Model Inference

The API reads the trained model's `feature_names_in_` and constructs the inference DataFrame in the exact feature order expected by the model. It then produces both the class prediction and the probability for class `1`.

### Interactive Documentation

When running locally, FastAPI provides:

```text
Swagger UI: http://127.0.0.1:8000/docs
OpenAPI schema: http://127.0.0.1:8000/openapi.json
```

### Run Locally

```bash
uvicorn src.serving_api:app --host 0.0.0.0 --port 8000
```

### Serving Deliverables

| Component | Location |
| :--- | :--- |
| FastAPI application | `src/serving_api.py` |
| Trained model artifact | `output/models/final_selected_model.joblib` |
| API documentation | `docs/task1_model_serving_api.md` |
| Docker configuration | `Dockerfile` |
| Dependencies | `requirements.txt` |

## Dataset

The project was initially referenced to the **Instacart Market Basket Analysis** dataset available on Kaggle.

The dataset contains information related to customers, orders, products, and order-product relationships, making it useful for studying purchasing behavior and demand patterns.

> **Note:** The dataset currently used in the project may differ from the dataset mentioned in the original project reference. Dataset details should be verified against the actual project data and implementation.

## Technologies Used

- **Python 3.11**
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computing
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning
- **Joblib** – Model serialization and loading
- **Jupyter Notebook** – Development and experimentation
- **MLflow** – Experiment tracking
- **FastAPI** – Model-serving REST API
- **Uvicorn** – ASGI server
- **Pydantic** – Request validation
- **Docker** – Application containerization
- **React** – Monitoring dashboard
- **Vite** – Frontend development server and build tool
- **Recharts** – Monitoring data visualizations
- **Redis** – Monitoring data storage
- **Kafka** – Prediction-event messaging/streaming
- **Git/GitHub** – Version control and project hosting

## Project Structure

The repository is organized approximately as follows:

```text
advanced-customer-data-analytics-platform/

│
├── dashboard/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── model_monitor.py
│   ├── serving_api.py
│   ├── compare_model_architectures.py
│   ├── tune_hyperparameters.py
│   ├── select_validate_model.py
│   └── ...
│
├── output/
│   └── models/
│       └── final_selected_model.joblib
│
├── docs/
│   ├── model_monitoring.md
│   ├── phase2_task1_model_training.md
│   ├── hyperparameter_tuning.md
│   ├── task5_evaluation_and_documentation.md
│   ├── task1_model_serving_api.md
│   └── ...
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
└── ...
```

The structure may evolve as additional project tasks and deliverables are completed.

## Installation

Clone the repository:

```bash
git clone https://github.com/yogeshs-codes/advanced-customer-data-analytics-platform.git
cd advanced-customer-data-analytics-platform
```

Create and activate a Python virtual environment if required:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

For the dashboard:

```powershell
cd dashboard
npm install
```

## Model Serving and Monitoring Setup

The model-serving API requires the trained model artifact to be available at:

```text
output/models/final_selected_model.joblib
```

From the project root, start the FastAPI application with:

```bash
uvicorn src.serving_api:app --host 0.0.0.0 --port 8000
```

The main API endpoints are:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/predict
http://127.0.0.1:8000/monitoring
```

For containerized execution, use the Docker commands described above.

## Usage

The project can be explored through the source code, notebooks, documentation, trained model artifacts, API implementation, monitoring service, and dashboard.

The overall workflow is:

```text
Problem Definition
        ↓
Data Collection & Understanding
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Development
        ↓
Model Architecture Comparison
        ↓
Hyperparameter Tuning
        ↓
Experiment Tracking
        ↓
Final Model Selection & Validation
        ↓
Final Model Evaluation
        ↓
FastAPI Model Serving
        ↓
Production Model Monitoring
        ↓
React Monitoring Dashboard
        ↓
Docker Containerization
        ↓
Prediction & Deployment Readiness
```

## Results

The project develops a customer-product purchase prediction solution using engineered behavioral features.

The final serving implementation exposes a selected **Gradient Boosting Classifier** model through FastAPI.

The API is designed to return:

- A binary future-purchase prediction.
- A human-readable prediction label.
- The probability of future purchase.
- The model version.
- The number of features used for inference.

The serving API also includes health monitoring, input validation, Swagger documentation, and Docker-based execution.

### Current Monitoring Results

During local monitoring validation, the `/monitoring` endpoint returned:

- **Total predictions:** 2
- **Positive predictions:** 2
- **Negative predictions:** 0
- **Positive prediction rate:** 100%
- **Average latency:** 96.11 ms
- **Minimum latency:** 71.73 ms
- **Maximum latency:** 120.50 ms
- **Average future-purchase probability:** approximately 52.9%
- **Latency anomaly:** false
- **Prediction distribution drift:** false
- **Active alerts:** none
- **Model version:** `gradient_boosting_v1`
- **Model algorithm:** `GradientBoostingClassifier`
- **Feature count:** 19

These values demonstrate that the monitoring service successfully collected prediction-level information and exposed aggregated operational metrics to the dashboard.

## Documentation

Important project documentation is maintained under the `docs/` directory.

Examples include:

- Model training documentation.
- Model architecture comparison.
- Hyperparameter tuning documentation.
- Final model selection and validation.
- Final model evaluation and documentation.
- Model serving API documentation.
- Model monitoring documentation.

The main monitoring documentation is:

```text
docs/model_monitoring.md
```

The model-serving documentation is:

```text
docs/task1_model_serving_api.md
```

## Implementation Evidence

The repository contains the actual implementation and supporting artifacts described above:

- `src/serving_api.py` contains the FastAPI application, Pydantic request validation, response schema, model loading, `/health`, `/predict`, and `/monitoring` endpoints.
- `src/model_monitor.py` contains the model monitoring implementation for prediction records, aggregated metrics, latency monitoring, prediction-distribution monitoring, and alerts.
- `dashboard/src/App.jsx` contains the React monitoring dashboard and its integration with the monitoring API.
- `dashboard/src/App.css` and `dashboard/src/index.css` provide the dashboard presentation and responsive layout.
- `docs/model_monitoring.md` documents the monitoring architecture, metrics, and operational behavior.
- `output/models/final_selected_model.joblib` is the trained model artifact used by the API.
- `docs/task1_model_serving_api.md` provides detailed Task 1 API documentation.
- `Dockerfile` defines the containerized API runtime.
- `requirements.txt` contains the Python application dependencies.
- `dashboard/package.json` defines the frontend dependencies, including React and Recharts.

These files provide direct source-code and configuration evidence for the model-serving and model-monitoring implementation.

## Version Control

Git is used to maintain the project history and track implementation changes.

The repository is hosted on GitHub:

```text
https://github.com/yogeshs-codes/advanced-customer-data-analytics-platform
```

Major project stages and implementation updates are committed to the repository so that the development process remains traceable.

## Future Improvements

Potential future improvements include:

- Improving feature engineering and feature selection.
- Evaluating additional machine learning algorithms.
- Improving model calibration and threshold selection.
- Adding automated API tests.
- Adding CI/CD validation for the serving application.
- Extending monitoring with additional production alerting rules.
- Adding automated monitoring reports and historical trend analysis.
- Deploying the monitoring dashboard alongside the production API.
- Deploying the containerized API and dashboard to a cloud platform.
- Automating model retraining and model versioning.
- Extending the serving layer for batch prediction.

## Author

**Yogesh S**

## License

This project is intended for educational and learning purposes.
