# Databricks notebook source
# MAGIC %run "/Workspace/Users/piyush.soni@hoonartek.com/test/customer_360/lake_base_connection"

# COMMAND ----------

paysim_bronze_df = spark.table("customer360.bronze.paysim_raw")

# COMMAND ----------

dim_account_df = spark.table("customer360.silver.dim_account")

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import *

source_account_lookup_df = (
    dim_account_df
    .select(
        col("account_sk").alias("source_account_sk"),
        col("account_number").alias("source_account_number")
    )
)

# COMMAND ----------

destination_account_lookup_df = (
    dim_account_df
    .select(
        col("account_sk").alias("destination_account_sk"),
        col("account_number").alias("destination_account_number")
    )
)

# COMMAND ----------

fact_transactions_df = (
    paysim_bronze_df
    .join(
        source_account_lookup_df,
        paysim_bronze_df.nameOrig ==
        source_account_lookup_df.source_account_number,
        "left"
    )
)

# COMMAND ----------

fact_transactions_df = (
    fact_transactions_df
    .join(
        destination_account_lookup_df,
        fact_transactions_df.nameDest ==
        destination_account_lookup_df.destination_account_number,
        "left"
    )
)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

fact_transactions_df = (
    fact_transactions_df
    .select(

        col("step").alias("transaction_step"),

        col("type").alias("transaction_type"),

        round(col("amount"), 2).alias("transaction_amount"),

        col("source_account_sk"),

        col("destination_account_sk"),

        col("nameOrig").alias("source_account_number"),

        col("nameDest").alias("destination_account_number"),

        round(
            col("oldbalanceOrg"), 2
        ).alias("source_old_balance"),

        round(
            col("newbalanceOrig"), 2
        ).alias("source_new_balance"),

        round(
            col("oldbalanceDest"), 2
        ).alias("destination_old_balance"),

        round(
            col("newbalanceDest"), 2
        ).alias("destination_new_balance"),

        col("isFraud").alias("fraud_flag"),

        col("isFlaggedFraud").alias("flagged_fraud_flag")
    )
)

# COMMAND ----------

window_spec_transaction = Window.orderBy(
    monotonically_increasing_id()
)

fact_transactions_df = (
    fact_transactions_df
    .withColumn(
        "transaction_sk",
        row_number().over(window_spec_transaction)
    )
)

# COMMAND ----------

fact_transactions_df = (
    fact_transactions_df
    .withColumn(
        "transaction_timestamp",
        expr(
            "timestampadd(HOUR, transaction_step, current_timestamp())"
        )
    )
)

# COMMAND ----------

fact_transactions_df = (
    fact_transactions_df

    .withColumn(
        "record_source",
        lit("PAYSIM_TRANSACTION_SYSTEM")
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

fact_transactions_df = (
    fact_transactions_df.select(

        "transaction_sk",

        "transaction_timestamp",

        "transaction_step",

        "transaction_type",

        "transaction_amount",

        "source_account_sk",

        "destination_account_sk",

        "source_account_number",

        "destination_account_number",

        "source_old_balance",

        "source_new_balance",

        "destination_old_balance",

        "destination_new_balance",

        "fraud_flag",

        "flagged_fraud_flag",

        "record_source",

        "created_timestamp",

        "updated_timestamp"
    )
)

# COMMAND ----------

display(fact_transactions_df)

# COMMAND ----------

fact_transactions_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer360.silver.fact_transactions")

# COMMAND ----------

(
    fact_transactions_df.write
    .format("postgresql")
    .option("host", host)
    .option("port", port)
    .option("database", database)
    .option("dbtable", "customer360.silver_fact_transactions")
    .option("user", "piyush.soni@hoonartek.com")
    .option("password", jwt_token)
    .mode("overwrite")
    .save()
)
 
print("Data loaded successfully")
