# Databricks notebook source
# MAGIC %run "/Workspace/Users/piyush.soni@hoonartek.com/test/customer_360/lake_base_connection"

# COMMAND ----------

paysim_bronze_df = spark.table("customer360.bronze.paysim_raw")

# COMMAND ----------

from pyspark.sql.functions import col

source_accounts_df = (
    paysim_bronze_df
    .select(
        col("nameOrig").alias("account_number")
    )
)

# COMMAND ----------

destination_accounts_df = (
    paysim_bronze_df
    .select(
        col("nameDest").alias("account_number")
    )
)

# COMMAND ----------

all_accounts_df = (
    source_accounts_df
    .union(destination_accounts_df)
    .distinct()
)   

# COMMAND ----------

display(all_accounts_df.limit(20))

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

dim_account_df = (
    all_accounts_df

    .withColumn(
        "account_type",
        when(
            col("account_number").startswith("C"),
            "CUSTOMER_ACCOUNT"
        ).when(
            col("account_number").startswith("M"),
            "MERCHANT_ACCOUNT"
        ).otherwise(
            "UNKNOWN"
        )
    )

    .withColumn(
        "account_status",
        lit("ACTIVE")
    )

    .withColumn(
        "account_open_date",
        current_date()
    )
)

# COMMAND ----------

window_spec_account = Window.orderBy("account_number")

dim_account_df = (
    dim_account_df
    .withColumn(
        "account_sk",
        row_number().over(window_spec_account)
    )
)

# COMMAND ----------

dim_account_df = dim_account_df.select(
    "account_sk",
    "account_number",
    "account_type",
    "account_status",
    "account_open_date"
)

# COMMAND ----------

display(dim_account_df)

# COMMAND ----------

dim_account_df = (
    dim_account_df
    .withColumn(
        "record_source",
        lit("PAYSIM_DATASET")
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

dim_account_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer360.silver.dim_account")

# COMMAND ----------

dim_account_df = dim_account_df.limit(500)

# COMMAND ----------

(
    dim_account_df.write
    .format("postgresql")
    .option("host", host)
    .option("port", port)
    .option("database", database)
    .option("dbtable", "customer360.silver_dim_account")
    .option("user", "piyush.soni@hoonartek.com")
    .option("password", jwt_token)
    .mode("overwrite")
    .save()
)
 
print("Data loaded successfully")
