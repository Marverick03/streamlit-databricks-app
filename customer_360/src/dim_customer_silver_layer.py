# Databricks notebook source
# MAGIC %run "/Workspace/Users/piyush.soni@hoonartek.com/test/customer_360/lake_base_connection"

# COMMAND ----------

customers_bronze_df = spark.table("customer360.bronze.customers_raw")

# COMMAND ----------

customers_bronze_df.display()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

dim_customer_df = (
    customers_bronze_df
    .select(
        col("customer_id"),

        col("vintage").alias("customer_tenure_months"),

        col("age"),

        lower(col("gender")).alias("gender"),

        col("dependents"),

        initcap(col("occupation")).alias("occupation"),

        col("city"),

        col("customer_nw_category").alias("net_worth_category"),

        col("branch_code"),

        round(col("current_balance"), 2).alias("current_balance"),

        round(
            col("previous_month_end_balance"), 2
        ).alias("previous_month_end_balance"),

        round(
            col("average_monthly_balance_prevQ"), 2
        ).alias("avg_balance_prev_quarter"),

        round(
            col("average_monthly_balance_prevQ2"), 2
        ).alias("avg_balance_prev_quarter_2"),

        round(
            col("current_month_credit"), 2
        ).alias("current_month_credit"),

        round(
            col("previous_month_credit"), 2
        ).alias("previous_month_credit"),

        round(
            col("current_month_debit"), 2
        ).alias("current_month_debit"),

        round(
            col("previous_month_debit"), 2
        ).alias("previous_month_debit"),

        round(
            col("current_month_balance"), 2
        ).alias("current_month_balance"),

        round(
            col("previous_month_balance"), 2
        ).alias("previous_month_balance"),

        col("churn").alias("churn_flag"),

        # FIX: Replaced to_timestamp with try_cast to handle 'NaT' strings safely
        expr("try_cast(last_transaction as timestamp)").alias("last_transaction_timestamp")
    )
)

# COMMAND ----------

dim_customer_df.display()

# COMMAND ----------

window_spec = Window.orderBy("customer_id")

dim_customer_df = (
    dim_customer_df
    .withColumn(
        "customer_sk",
        row_number().over(window_spec)
    )
)

# COMMAND ----------

dim_customer_df = dim_customer_df.select(
    "customer_sk",
    "customer_id",
    "customer_tenure_months",
    "age",
    "gender",
    "dependents",
    "occupation",
    "city",
    "net_worth_category",
    "branch_code",
    "current_balance",
    "previous_month_end_balance",
    "avg_balance_prev_quarter",
    "avg_balance_prev_quarter_2",
    "current_month_credit",
    "previous_month_credit",
    "current_month_debit",
    "previous_month_debit",
    "current_month_balance",
    "previous_month_balance",
    "churn_flag",
    "last_transaction_timestamp"
)

# COMMAND ----------

dim_customer_df.select(
    countDistinct("customer_id")
).show()

# COMMAND ----------

dim_customer_df.display()

# COMMAND ----------

dim_customer_df = (
    dim_customer_df
    .withColumn(
        "record_source",
        lit("BANK_CUSTOMER_DATASET")
    )
    .withColumn(
        "created_timestamp",
        current_timestamp()
    )
    .withColumn(
        "updated_timestamp",
        current_timestamp()
    )
)

# COMMAND ----------

dim_customer_df.display()

# COMMAND ----------

dim_customer_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer360.silver.dim_customer")

# COMMAND ----------

dim_customer_df = dim_customer_df.limit(500)

# COMMAND ----------

# DBTITLE 1,writing in postgres
(
    dim_customer_df.write
    .format("postgresql")
    .option("host", host)
    .option("port", port)
    .option("database", database)
    .option("dbtable", "customer360.silver_dim_customer")
    .option("user", "piyush.soni@hoonartek.com")
    .option("password", jwt_token)
    .mode("overwrite")
    .save()
)
 
print("Data loaded successfully")

# COMMAND ----------

customers_bronze_df = spark.table("customer360.bronze.customers_raw")

# COMMAND ----------

customers_bronze_df.printSchema()

# COMMAND ----------

paysim_bronze_df = spark.table("customer360.bronze.paysim_raw")

# COMMAND ----------

paysim_bronze_df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

dim_customer_df = (
    customers_bronze_df
    .select(
        col("Customer_Id").alias("customer_id"),
        col("Gender").alias("gender"),
        col("Age").alias("age"),
        col("Balance").alias("balance"),
        col("EstimatedSalary").alias("estimated_salary"),
        col("Tenure").alias("tenure"),
        col("IsActiveMember").alias("active_flag"),
        col("Exited").alias("churn_flag")
    )
)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window


window_spec = Window.orderBy("customer_id")

dim_customer_df = (
    dim_customer_df
    .withColumn(
        "customer_sk",
        row_number().over(window_spec)
    )
)

# COMMAND ----------

display(dim_customer_df)
