# Databricks notebook source
# DBTITLE 1,Notebook Header
# Databricks notebook source
# MAGIC %md
# MAGIC # Insurance Propensity Model Training
# MAGIC 
# MAGIC This notebook trains a logistic regression model to predict insurance propensity.
# MAGIC 
# MAGIC **Parameterized for MLOps bundle deployment**

# COMMAND ----------

# DBTITLE 1,Setup Parameters
# Define widgets for parameterization
dbutils.widgets.text("catalog", "customer360", "UC Catalog")
dbutils.widgets.text("schema", "gold", "UC Schema (data)")
dbutils.widgets.text("model_catalog", "customer360", "Model Catalog")
dbutils.widgets.text("model_schema", "ml_models", "Model Schema")
dbutils.widgets.text("model_name", "insurance_propensity_model", "Model Name")
dbutils.widgets.text("experiment_path", "/Shared/mlops/insurance_propensity", "MLflow Experiment Path")
dbutils.widgets.text("env", "dev", "Environment (dev/prod)")

# Get widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_catalog = dbutils.widgets.get("model_catalog")
model_schema = dbutils.widgets.get("model_schema")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
env = dbutils.widgets.get("env")

print(f"Training Insurance Propensity Model")
print(f"Data Source: {catalog}.{schema}.customer_360")
print(f"Model Target: {model_catalog}.{model_schema}.{model_name}")
print(f"Environment: {env}")

# COMMAND ----------

# DBTITLE 1,Import Libraries and Setup MLflow
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import when, col
from pyspark.ml.functions import vector_to_array
from mlflow.models.signature import infer_signature

# Set MLflow tracking to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)

# COMMAND ----------

# DBTITLE 1,Load Data
# Load customer data
customer360_df = spark.table(f"{catalog}.{schema}.customer_360")
print(f"Loaded {customer360_df.count()} records")

# COMMAND ----------

# DBTITLE 1,Create Target Variable
# Create target variable (insurance_v1) based on business rules
insurance_df = (
    customer360_df
    .withColumn(
        "insurance_v1",
        when(
            (col("age") >= 45) &
            (col("dependents") >= 1) &
            (col("current_balance") >= 3000),
            1
        ).otherwise(0)
    )
)

print(f"Target distribution:")
insurance_df.groupBy("insurance_v1").count().show()

# COMMAND ----------

# DBTITLE 1,Feature Selection
# Define feature columns
feature_columns = [
    "customer_tenure_months",
    "net_worth_category",
    "avg_balance_prev_quarter",
    "total_transactions",
    "total_transaction_amount",
    "avg_transaction_amount",
    "engagement_score"
]

# Select features and label
training_df = insurance_df.select(
    "customer_id",
    *feature_columns,
    "insurance_v1"
)

print(f"Features: {feature_columns}")

# COMMAND ----------

# DBTITLE 1,Train/Test Split
# Split data into train and test sets
train_df, test_df = training_df.randomSplit([0.8, 0.2], seed=42)

print(f"Train records: {train_df.count()}")
print(f"Test records: {test_df.count()}")

# COMMAND ----------

# DBTITLE 1,Build ML Pipeline
# Build ML Pipeline
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="insurance_v1",
    maxIter=100
)

pipeline = Pipeline(stages=[assembler, scaler, lr])
print("Pipeline created: VectorAssembler → StandardScaler → LogisticRegression")

# COMMAND ----------

# DBTITLE 1,Setup UC Model Registry
# Ensure UC schema and volume exist for model artifacts
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {model_catalog}.{model_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {model_catalog}.{model_schema}.mlflow_tmp")

DFS_TMP_DIR = f"/Volumes/{model_catalog}/{model_schema}/mlflow_tmp"
print(f"Model artifacts will be staged in: {DFS_TMP_DIR}")

# COMMAND ----------

# DBTITLE 1,Train and Register Model
# Train model with MLflow tracking
full_model_name = f"{model_catalog}.{model_schema}.{model_name}"

with mlflow.start_run(run_name=f"insurance_propensity_{env}") as run:
    # Train the pipeline
    pipeline_model = pipeline.fit(train_df)
    
    # Make predictions on test set
    predictions = pipeline_model.transform(test_df)
    
    # Evaluate model
    evaluator = BinaryClassificationEvaluator(
        labelCol="insurance_v1",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    
    print(f"Model AUC: {auc:.4f}")
    
    # Log parameters and metrics
    mlflow.log_param("features", feature_columns)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("environment", env)
    mlflow.log_param("data_source", f"{catalog}.{schema}.customer_360")
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("train_records", train_df.count())
    mlflow.log_metric("test_records", test_df.count())
    
    # Create signature for model input/output schema
    input_sample = train_df.select(*feature_columns).limit(10).toPandas()
    output_sample = pipeline_model.transform(train_df.limit(10)).select("prediction").toPandas()
    signature = infer_signature(input_sample, output_sample)
    
    # Log and register model to Unity Catalog
    mlflow.spark.log_model(
        pipeline_model,
        artifact_path="model",
        registered_model_name=full_model_name,
        dfs_tmpdir=DFS_TMP_DIR,
        signature=signature
    )
    
    run_id = run.info.run_id
    print(f"Model registered: {full_model_name}")
    print(f"Run ID: {run_id}")

# COMMAND ----------

# DBTITLE 1,Score Population
# Score entire population
all_predictions = pipeline_model.transform(training_df)

# Extract probability scores
insurance_scores = (
    all_predictions
    .withColumn(
        "probability_array",
        vector_to_array("probability")
    )
    .select(
        "customer_id",
        col("probability_array")[1].alias("insurance_score")
    )
)

print(f"Generated {insurance_scores.count()} insurance propensity scores")

# COMMAND ----------

# DBTITLE 1,Save Scores
# Save scores to gold table
output_table = f"{catalog}.{schema}.insurance_scores_test"

insurance_scores.write \
    .mode("overwrite") \
    .saveAsTable(output_table)

print(f"Scores saved to: {output_table}")
print("Training completed successfully! ✓")

# COMMAND ----------


