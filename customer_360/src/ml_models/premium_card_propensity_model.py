# Databricks notebook source
# DBTITLE 1,Notebook Header
# Databricks notebook source
# MAGIC %md
# MAGIC # Premium Card Propensity Model Training
# MAGIC 
# MAGIC This notebook trains a logistic regression model to predict premium card propensity.
# MAGIC 
# MAGIC **Parameterized for MLOps bundle deployment**
# MAGIC 
# MAGIC **Note**: This follows the same structure as insurance_propensity_model.py.
# MAGIC Adjust the target variable creation logic and feature columns as needed for premium card.

# COMMAND ----------

# DBTITLE 1,Setup Parameters
# Define widgets for parameterization
dbutils.widgets.text("catalog", "customer360", "UC Catalog")
dbutils.widgets.text("schema", "gold", "UC Schema (data)")
dbutils.widgets.text("model_catalog", "customer360", "Model Catalog")
dbutils.widgets.text("model_schema", "ml_models", "Model Schema")
dbutils.widgets.text("model_name", "premium_card_propensity_model", "Model Name")
dbutils.widgets.text("experiment_path", "/Shared/mlops/premium_card_propensity", "MLflow Experiment Path")
dbutils.widgets.text("env", "dev", "Environment (dev/prod)")

# Get widget values
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_catalog = dbutils.widgets.get("model_catalog")
model_schema = dbutils.widgets.get("model_schema")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
env = dbutils.widgets.get("env")

print(f"Training Premium Card Propensity Model")
print(f"Data Source: {catalog}.{schema}.customer_360")
print(f"Model Target: {model_catalog}.{model_schema}.{model_name}")
print(f"Environment: {env}")

# COMMAND ----------

# DBTITLE 1,Implementation Notes
# MAGIC %md
# MAGIC ### TODO: Complete implementation
# MAGIC 
# MAGIC Copy the training logic from insurance_propensity_model.py and adjust:
# MAGIC 1. Target variable creation logic for premium card
# MAGIC 2. Feature columns specific to premium card propensity
# MAGIC 3. Output table name: `premium_scores`

# COMMAND ----------

# DBTITLE 1,Load Data
# Load customer360 data
customer360_df = spark.table(f"{catalog}.{schema}.customer_360")
print(f"Loaded {customer360_df.count()} customer records")

# COMMAND ----------

# DBTITLE 1,Create Target Variable
from pyspark.sql.functions import when, col

# Create target variable: premium_credit_card_v3
# Business Rule: Customers with high balance AND high engagement
ml_df = customer360_df.withColumn(
    "premium_credit_card_v3",
    when(
        (col("current_balance") > 10000) & (col("engagement_score") > 0.73),
        1
    ).otherwise(0)
)

print(f"\nTarget Distribution:")
ml_df.groupBy("premium_credit_card_v3").count().show()

# COMMAND ----------

# DBTITLE 1,Define Features
# Feature columns for Premium Card propensity model
feature_columns = [
    "customer_id",
    "customer_tenure_months",
    "age",
    "dependents",
    "net_worth_category",
    "current_month_credit",
    "previous_month_credit",
    "current_month_debit",
    "previous_month_debit",
    "total_transaction_amount"
]

print(f"\nNumber of features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# COMMAND ----------

# DBTITLE 1,Prepare Training Data
# Select feature columns and target variable
training_df = ml_df.select(
    *feature_columns,
    "premium_credit_card_v3"
)

print(f"\nTraining data shape: {training_df.count()} rows, {len(training_df.columns)} columns")

# COMMAND ----------

# DBTITLE 1,Handle Missing Values
# Handle missing values - fill dependents with 0
training_df = training_df.fillna({"dependents": 0})

# Verify no missing values
from pyspark.sql.functions import sum as spark_sum
null_counts = training_df.select(
    *[spark_sum(col(c).isNull().cast("int")).alias(c) for c in training_df.columns]
)
print("\nNull counts after fillna:")
null_counts.show()

# COMMAND ----------

# DBTITLE 1,Build ML Pipeline
# Build ML Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

# Create pipeline stages
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
    labelCol="premium_credit_card_v3",
    maxIter=100
)

# Assemble pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

print("\nPipeline stages:")
for i, stage in enumerate(pipeline.getStages()):
    print(f"  {i+1}. {type(stage).__name__}")

# COMMAND ----------

# DBTITLE 1,Train and Evaluate Model
# Split data into train and test
train_df, test_df = training_df.randomSplit([0.8, 0.2], seed=42)

print(f"\nTraining set size: {train_df.count()}")
print(f"Test set size: {test_df.count()}")

# Train the model
print("\nTraining Premium Card Propensity Model...")
model = pipeline.fit(train_df)
print("✅ Training complete!")

# Make predictions on test set
predictions = model.transform(test_df)

# Evaluate model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    labelCol="premium_credit_card_v3",
    rawPredictionCol="rawPrediction"
)

auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print(f"\n🎯 Premium Card Model Performance")
print(f"   AUC-ROC: {auc:.4f}")

# COMMAND ----------

# DBTITLE 1,MLflow Logging and Registration
# MLflow Tracking
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

# Setup MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)

# Create UC resources if needed
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {model_catalog}.{model_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {model_catalog}.{model_schema}.mlflow_tmp")

# Start MLflow run
with mlflow.start_run(run_name=f"premium_card_{env}") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("features", ",".join(feature_columns))
    mlflow.log_param("num_features", len(feature_columns))
    mlflow.log_param("train_size", train_df.count())
    mlflow.log_param("test_size", test_df.count())
    mlflow.log_param("environment", env)
    mlflow.log_param("catalog", catalog)
    mlflow.log_param("schema", schema)
    
    # Log metrics
    mlflow.log_metric("auc_roc", auc)
    
    # Infer signature
    input_sample = train_df.limit(10).toPandas()
    output_sample = predictions.select("prediction", "probability").limit(10).toPandas()
    signature = infer_signature(input_sample, output_sample)
    
    # Log model to MLflow and register to Unity Catalog
    model_info = mlflow.spark.log_model(
        model,
        artifact_path="model",
        registered_model_name=f"{model_catalog}.{model_schema}.{model_name}",
        dfs_tmpdir=f"/Volumes/{model_catalog}/{model_schema}/mlflow_tmp",
        signature=signature
    )
    
    print(f"\n✅ Model registered to Unity Catalog: {model_catalog}.{model_schema}.{model_name}")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Experiment: {experiment_path}")

# COMMAND ----------

# DBTITLE 1,Score Population and Save Results
# Score entire customer population
print("\nScoring all customers...")
predictions_all = model.transform(training_df)

# Extract probability scores
from pyspark.ml.functions import vector_to_array

premium_scores = (
    predictions_all
    .withColumn("probability_array", vector_to_array("probability"))
    .select(
        "customer_id",
        col("probability_array")[1].alias("premium_card_score"),
        "prediction",
        "premium_credit_card_v3"
    )
)

print(f"Scored {premium_scores.count()} customers")
print("\nTop 10 customers by premium card propensity:")
premium_scores.orderBy(col("premium_card_score").desc()).show(10, truncate=False)

# Save scores to gold layer
print(f"\nSaving scores to {catalog}.{schema}.premium_scores...")
premium_scores.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.premium_scores_test")
print("✅ Scores saved successfully!")

print(f"\n🎉 Premium Card Model Training Complete!")

# COMMAND ----------


