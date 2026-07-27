# ML Models - MLOps Training Pipelines

This directory contains parameterized ML model training notebooks for the Customer360 propensity models.

## 🎯 Models

### 1. Insurance Propensity Model
**File**: `insurance_propensity_model.py`  
**Status**: ✅ Fully implemented with MLflow integration  
**Target**: Predict customer propensity for insurance products  
**Output**: `customer360.gold.insurance_scores`

### 2. Personal Loan Propensity Model
**File**: `personal_loan_propensity_model.py`  
**Status**: ✅ Fully implemented with MLflow integration  
**Target**: Predict customer propensity for personal loans  
**Output**: `customer360.gold.personal_loan_scores`

### 3. Premium Card Propensity Model
**File**: `premium_card_propensity_model.py`  
**Status**: 🚧 Template created (needs implementation)  
**Target**: Predict customer propensity for premium credit cards  
**Output**: `customer360.gold.premium_scores`

### 4. Wealth Management Propensity Model
**File**: `wealth_management_propensity_model.py`  
**Status**: 🚧 Template created (needs implementation)  
**Target**: Predict customer propensity for wealth management services  
**Output**: `customer360.gold.wealth_scores`

---

## 📋 Notebook Parameters

All notebooks accept the following widget parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|----------|
| `catalog` | UC Catalog for data | `customer360` | `customer360` |
| `schema` | UC Schema for data | `gold` | `gold` |
| `model_catalog` | UC Catalog for models | `customer360` | `customer360` |
| `model_schema` | UC Schema for models | `ml_models` | `ml_models` |
| `model_name` | Model name | (varies) | `insurance_propensity_model` |
| `experiment_path` | MLflow experiment path | (varies) | `/Shared/mlops/dev/insurance` |
| `env` | Environment | `dev` | `dev` or `prod` |

---

## 🔄 Workflow

### Training Pipeline

```
1. Load Data
   └─ Read from customer360.gold.customer_360

2. Feature Engineering
   ├─ Create target variable (business rules)
   ├─ Select feature columns
   └─ Handle missing values

3. Build ML Pipeline
   ├─ VectorAssembler (combine features)
   ├─ StandardScaler (scale features)
   └─ LogisticRegression (train model)

4. Train & Evaluate
   ├─ 80/20 train/test split
   ├─ Fit pipeline on train data
   └─ Evaluate on test data (AUC-ROC)

5. MLflow Tracking
   ├─ Log parameters (features, model_type, env)
   ├─ Log metrics (AUC, train/test counts)
   └─ Log model with signature

6. Register to UC
   └─ Register model to Unity Catalog

7. Score Population
   ├─ Score all customers
   └─ Extract probability scores

8. Save Results
   └─ Write scores to gold layer table
```

---

## 🚀 Running Models

### Option 1: Via Bundle Job (Recommended)

```bash
# Deploy the bundle
databricks bundle deploy --target dev

# Run the ML training pipeline (all 4 models in parallel)
databricks bundle run ml_training_pipeline --target dev
```

### Option 2: Manual Execution

Open any notebook and run with default parameters, or override:

```python
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("catalog", "customer360")
# ... then run all cells
```

### Option 3: Programmatic Execution

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

w.jobs.run_now(
    job_id=<ml_training_pipeline_job_id>,
    notebook_params={
        "env": "prod",
        "catalog": "customer360"
    }
)
```

---

## 📊 MLflow Integration

### Experiment Organization

```
/Shared/mlops/
├── dev/
│   ├── insurance_propensity/
│   ├── personal_loan_propensity/
│   ├── premium_card_propensity/
│   └── wealth_management_propensity/
└── prod/
    ├── insurance_propensity/
    ├── personal_loan_propensity/
    ├── premium_card_propensity/
    └── wealth_management_propensity/
```

### Model Registry (Unity Catalog)

```
customer360.ml_models/
├── insurance_propensity_model (v1, v2, ...)
├── personal_loan_propensity_model (v1, v2, ...)
├── premium_card_propensity_model
└── wealth_management_propensity_model
```

### Artifacts Storage

Spark ML artifacts are staged in:
```
/Volumes/customer360/ml_models/mlflow_tmp/
```

---

## 🛠️ Completing the Templates

For **premium_card_propensity_model.py** and **wealth_management_propensity_model.py**:

1. Copy the full implementation from `insurance_propensity_model.py`
2. Update the target variable creation logic (cell 5)
3. Update the feature columns list (cell 6)
4. Update the output table name in the last cell
5. Test the notebook manually first
6. Then run via the bundle job

---

## 📈 Next Steps

1. ✅ Complete template notebooks (premium_card, wealth_management)
2. ⏭️ Add model monitoring (data drift, performance tracking)
3. ⏭️ Implement automated retraining triggers
4. ⏭️ Add model explainability (SHAP values)
5. ⏭️ Set up model deployment to serving endpoints
6. ⏭️ Implement Champion/Challenger A/B testing
7. ⏭️ Add feature engineering pipeline

---

## 📞 Support

For questions or issues, contact:
* **Email**: rushikesh.kumavat@hoonartek.com
* **Team**: Data Science / MLOps