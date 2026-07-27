# Databricks notebook source
# MAGIC %run "/Workspace/Users/piyush.soni@hoonartek.com/test/customer_360/lake_base_connection"

# COMMAND ----------

dim_customer_df = spark.table("customer360.silver.dim_customer")

# COMMAND ----------

dim_account_df = spark.table("customer360.silver.dim_account")

# COMMAND ----------

customer_map_df = (
    dim_customer_df
    .select(
        "customer_sk",
        "customer_id"
    )
)

# COMMAND ----------

account_map_df = (
    dim_account_df
    .select(
        "account_sk",
        "account_number",
        "account_type"
    )
)

# COMMAND ----------

account_map_df.display()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import *

customer_map_df = (
    customer_map_df
    .withColumn(
        "mapping_id",
        row_number().over(
            Window.orderBy("customer_sk")
        )
    )
)


# COMMAND ----------


from pyspark.sql.functions import row_number
from pyspark.sql.window import Window

account_map_df = (
    account_map_df
    .withColumn(
        "mapping_id",
        row_number().over(
            Window.orderBy("account_sk")
        )
    )
)

# COMMAND ----------

account_map_df.display()

# COMMAND ----------

bridge_customer_account_df = (
    account_map_df
    .join(
        customer_map_df,
        on="mapping_id",
        how="left"
    )
)

# COMMAND ----------

bridge_customer_account_df = (
    bridge_customer_account_df

    .withColumn(
        "relationship_type",
        lit("PRIMARY_OWNER")
    )

    .withColumn(
        "relationship_start_date",
        current_date()
    )

    .withColumn(
        "relationship_end_date",
        lit(None).cast("date")
    )

    .withColumn(
        "active_relationship_flag",
        lit(1)
    )
)

# COMMAND ----------

bridge_customer_account_df = (
    bridge_customer_account_df.select(
        "customer_sk",
        "customer_id",
        "account_sk",
        "account_number",
        "account_type",
        "relationship_type",
        "relationship_start_date",
        "relationship_end_date",
        "active_relationship_flag"
    )
)

# COMMAND ----------

bridge_customer_account_df.display()

# COMMAND ----------

bridge_customer_account_df = (
    bridge_customer_account_df

    .withColumn(
        "record_source",
        lit("CUSTOMER_ACCOUNT_MAPPING")
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

bridge_customer_account_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer360.silver.bridge_customer_account")

# COMMAND ----------

(
    bridge_customer_account_df.write
    .format("postgresql")
    .option("host", host)
    .option("port", port)
    .option("database", database)
    .option("dbtable", "customer360.silver_bridge_customer_account")
    .option("user", "piyush.soni@hoonartek.com")
    .option("password", jwt_token)
    .mode("overwrite")
    .save()
)
 
print("Data loaded successfully")
