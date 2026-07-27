# Databricks notebook source
# MAGIC %run "/Workspace/Users/piyush.soni@hoonartek.com/test/customer_360/lake_base_connection"

# COMMAND ----------

customers_raw_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("/Volumes/customer360/filestore/manual_files_customer_360/customers.csv")
)

# COMMAND ----------

customers_raw_df.display()

# COMMAND ----------

customers_raw_df.count()

# COMMAND ----------

# DBTITLE 1,Writing to delta
customers_raw_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer360.bronze.customers_raw")

# COMMAND ----------

# customers_raw_df = customers_raw_df.limit(500)

# COMMAND ----------

# DBTITLE 1,Writing to lakebase
(
    customers_raw_df.write
    .format("postgresql")
    .option("host", host)
    .option("port", port)
    .option("database", database)
    .option("dbtable", "customer360.bronze_customers_raw")
    .option("user", user)
    .option("password", jwt_token)
    .mode("overwrite")
    .save()
)

print("Data loaded successfully")

# COMMAND ----------

paysim_raw_df = (
    spark.read
    .format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load("/Volumes/customer360/filestore/manual_files_customer_360/paysim.csv")
)

# COMMAND ----------

paysim_raw_df.display()

# COMMAND ----------

paysim_raw_df.count()

# COMMAND ----------

paysim_raw_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer360.bronze.paysim_raw")

# COMMAND ----------

paysim_raw_df = paysim_raw_df.limit(500)

# COMMAND ----------

(
    paysim_raw_df.write
    .format("postgresql")
    .option("host", host)
    .option("port", port)
    .option("database", database)
    .option("dbtable", "customer360.bronze_paysim_raw")
    .option("user", user)
    .option("password", jwt_token)
    .mode("overwrite")
    .save()
)

print("Data loaded successfully")
