# Advanced Customer Data Analytics Platform

## Overview

This project focuses on analyzing customer purchasing behavior and building a machine learning solution to understand and predict customer demand.

The project follows a structured end-to-end machine learning workflow, covering problem definition, data understanding, preprocessing, exploratory data analysis, feature engineering, model development, model evaluation, experiment tracking, model selection, and model serving.

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

11. **Containerization**
    - Package the model-serving application with Docker.
    - Install application dependencies inside the container.
    - Include the trained model required for inference.
    - Run the API through Uvicorn on port 8000.

12. **Insights & Conclusion**
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

### Phase 3 - FastAPI and Docker
### Task 1 - Model Serving API

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

The API serves the selected Gradient Boosting Classifier model using the trained model artifact:

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
- **Git/GitHub** – Version control and project hosting

## Project Structure

The repository is organized approximately as follows:

```text
advanced-customer-data-analytics-platform/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── ...
│
├── src/
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

## Model Serving Setup

The model-serving API requires the trained model artifact to be available at:

```text
output/models/final_selected_model.joblib
```

From the project root, the FastAPI application can be started with:

```bash
uvicorn src.serving_api:app --host 0.0.0.0 --port 8000
```

For containerized execution, use the Docker commands described above.

## Usage

The project can be explored through the source code, notebooks, documentation, trained model artifacts, and API implementation provided in the repository.

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

Detailed model evaluation and task-specific results are documented in the relevant files under `docs/`.

## Documentation

Important project documentation is maintained under the `docs/` directory.

Examples include:

- Model training documentation.
- Model architecture comparison.
- Hyperparameter tuning documentation.
- Final model selection and validation.
- Final model evaluation and documentation.
- Model serving API documentation.

The model-serving documentation is:

```text
docs/task1_model_serving_api.md
```

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
- Adding API monitoring and logging.
- Deploying the containerized API to a cloud platform.
- Adding an interactive dashboard for business users.
- Automating model retraining and model versioning.
- Extending the serving layer for batch prediction.

## Author

**Yogesh S**

## License

This project is intended for educational and learning purposes.
