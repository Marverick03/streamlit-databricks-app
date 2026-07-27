# Databricks notebook source
# MAGIC %run "/Workspace/Users/piyush.soni@hoonartek.com/test/customer_360/lake_base_connection"

# COMMAND ----------

dim_customer_df = spark.table("customer360.silver.dim_customer")

# COMMAND ----------

bridge_customer_account_df = spark.table(
    "customer360.silver.bridge_customer_account"
)

# COMMAND ----------

fact_transactions_df = spark.table(
    "customer360.silver.fact_transactions"
)

# COMMAND ----------

customer_account_df = (
    bridge_customer_account_df
    .select(
        "customer_sk",
        "account_sk"
    )
)

# COMMAND ----------

customer_transactions_df = (
    customer_account_df
    .join(
        fact_transactions_df,
        customer_account_df.account_sk ==
        fact_transactions_df.source_account_sk,
        "inner"
    )
)

# COMMAND ----------

from pyspark.sql.functions import *

customer_transaction_metrics_df = (
    customer_transactions_df
    .groupBy("customer_sk")

    .agg(

        count("transaction_sk")
        .alias("total_transactions"),

        round(
            sum("transaction_amount"), 2
        ).alias("total_transaction_amount"),

        round(
            avg("transaction_amount"), 2
        ).alias("avg_transaction_amount"),

        round(
            max("transaction_amount"), 2
        ).alias("max_transaction_amount"),

        round(
            min("transaction_amount"), 2
        ).alias("min_transaction_amount"),

        sum("fraud_flag")
        .alias("fraud_transaction_count"),

        sum("flagged_fraud_flag")
        .alias("flagged_fraud_count")
    )
)

# COMMAND ----------

customer_account_metrics_df = (
    bridge_customer_account_df
    .groupBy("customer_sk")

    .agg(
        countDistinct("account_sk")
        .alias("total_accounts")
    )
)

# COMMAND ----------

customer_transaction_metrics_df = (
    customer_transaction_metrics_df

    .withColumn(
        "fraud_transaction_ratio",
        round(
            col("fraud_transaction_count") /
            col("total_transactions"),
            4
        )
    )
)

# COMMAND ----------

customer_360_df = (
    dim_customer_df

    .join(
        customer_account_metrics_df,
        on="customer_sk",
        how="left"
    )

    .join(
        customer_transaction_metrics_df,
        on="customer_sk",
        how="left"
    )
)

# COMMAND ----------

customer_360_df = (
    customer_360_df.fillna({

        "total_accounts": 0,

        "total_transactions": 0,

        "total_transaction_amount": 0,

        "avg_transaction_amount": 0,

        "max_transaction_amount": 0,

        "min_transaction_amount": 0,

        "fraud_transaction_count": 0,

        "flagged_fraud_count": 0,

        "fraud_transaction_ratio": 0
    })
)

# COMMAND ----------

# customer_360_df = (
#     customer_360_df

#     .withColumn(

#         "customer_segment",

#         when(
#             col("total_transaction_amount") >= 1000000,
#             "HIGH_VALUE"
#         ).when(
#             col("total_transaction_amount") >= 250000,
#             "MID_VALUE"
#         ).otherwise(
#             "LOW_VALUE"
#         )
#     )
# )

# COMMAND ----------

customer_360_df = (
    customer_360_df

    .withColumn(

        "engagement_score",

        round(

            (
                col("total_transactions") * 0.4
                +
                col("total_accounts") * 0.3
                +
                col("current_balance") * 0.00001
            ),

            2
        )
    )
)

# COMMAND ----------

customer_360_df = (
    customer_360_df

    .withColumn(
        "record_source",
        lit("CUSTOMER_360_AGGREGATION")
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

customer_360_df.createOrReplaceTempView("customer_360")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESC customer_360

# COMMAND ----------

customer_360_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer360.gold.customer_360")

# COMMAND ----------

customer_360_df = customer_360_df.limit(500)

# COMMAND ----------

# (
#     customer_360_df.write
#     .format("postgresql")
#     .option("host", host)
#     .option("port", port)
#     .option("database", database)
#     .option("dbtable", "customer360.gold_customer_360")
#     .option("user", "piyush.soni@hoonartek.com")
#     .option("password", jwt_token)
#     .mode("overwrite")
#     .save()
# )
 
# print("Data loaded successfully")
