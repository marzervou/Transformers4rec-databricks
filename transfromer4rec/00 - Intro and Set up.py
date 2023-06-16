# Databricks notebook source
# MAGIC  %md
# MAGIC This notebook perform all the set up before the data preprocessing step. The dataset we will use is the `YOOCHOOSE` dataset which contains a collection of sessions from a retailer. Each session  encapsulates the click events that the user performed in that session. The data was collected during several months in the year of 2014, reflecting the clicks and purchases performed by the users of an on-line retailer in Europe
# MAGIC
# MAGIC The dataset is available on [Kaggle](https://www.kaggle.com/chadgostopp/recsys-challenge-2015). You need to download it and copy to the `DATA_FOLDER` path. 

# COMMAND ----------

KAGGLE_USERNAME = dbutils.secrets.get(scope="mz-scopes", key="kaggle_username")
KAGGLE_KEY = dbutils.secrets.get(scope="mz-scopes", key="kaggle_key")

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip install kaggle
# MAGIC export KAGGLE_USERNAME=KAGGLE_USERNAME
# MAGIC export KAGGLE_KEY=KAGGLE_KEY
# MAGIC kaggle datasets download -d chadgostopp/recsys-challenge-2015 -p /dbfs/merlin/data/
# MAGIC cd /dbfs/merlin/data/ &&  unzip /dbfs/merlin/data/recsys-challenge-2015.zip 
# MAGIC

# COMMAND ----------

# MAGIC  %md Note that we are only using the `yoochoose-clicks.dat` file.

# COMMAND ----------

import os 
from pyspark.sql.types import *

DATA_FOLDER = "dbfs:/merlin/data/" #where the raw data is stored
FILENAME_PATTERN = 'yoochoose-clicks.dat' #which files to read
DATA_PATH = os.path.join(DATA_FOLDER, FILENAME_PATTERN) 

#  reate table schema to read it with spark
schema = StructType([\
    StructField("session_id", IntegerType(), True),\
    StructField("timestamp", TimestampType(), True),\
    StructField("item_id", IntegerType(), True),\
    StructField("category", IntegerType(), True)])

#read the data
df = spark.read.format("csv").schema(schema).option("header", "false").load(DATA_PATH)

# COMMAND ----------

# MAGIC %md Let's ue the dataprofiler available in databricsk to get an understanding of the data

# COMMAND ----------

df.display()

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# MAGIC %md We are writing the data into a Delta table - our raw/bronze layer

# COMMAND ----------

RAW_FOLDER = "dbfs:/merlin/data/raw" #where the raw data is stored

# writing the file into output location - Default format Delta
df.write.mode("overwrite").save(RAW_FOLDER)

# COMMAND ----------


