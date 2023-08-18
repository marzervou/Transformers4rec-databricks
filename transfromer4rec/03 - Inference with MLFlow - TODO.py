# Databricks notebook source
# MAGIC %pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC %pip install mlflow==2.3

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Model from MLFlow
import mlflow
from pyspark.sql.functions import struct, col

logged_model = 'runs:/8853d64897314201859e6edeeecfe6b1/mz-tran4rec'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model =mlflow.pyfunc.load_model(model_uri=logged_model)

# COMMAND ----------

data_to_predict = "/FileStore/mzervou/test.parquet"

# COMMAND ----------

import pyspark.pandas as ps

df = spark.read.parquet(data_to_predict)

# COMMAND ----------

df.show()

# COMMAND ----------

df2 = df.select('item_id-list_seq', 'category-list_seq', 'product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq')

# COMMAND ----------

print(loaded_model.predict(df2.toPandas().to_numpy()))

# COMMAND ----------



# COMMAND ----------

predict_df = nvt.Dataset("/Workspace/Repos/maria.zervou@databricks.com/Transformers4rec-databricks/transfromer4rec/dbfs:/merlin/data/processed/preproc_sessions_by_day/3/test.parquet")

# COMMAND ----------

# Get a merlin dataset from a set of parquet files
import merlin.io
dataset = merlin.io.Dataset("/Workspace/Repos/maria.zervou@databricks.com/Transformers4rec-databricks/transfromer4rec/dbfs:/merlin/data/processed/preproc_sessions_by_day/3", engine="parquet")

# Create a torch dataloader from the dataset, loading 65K items
# per batch
from merlin.dataloader.torch import Loader
loader = Loader(dataset, batch_size=10)

# Get a single batch of data
inputs = next(loader)

# COMMAND ----------

inputs

# COMMAND ----------

inputs

# COMMAND ----------

import transformers4rec
train_loader = transformers4rec.torch.utils.data_utils.MerlinDataLoader.from_schema(
        schema,
        paths_or_dataset="/Workspace/Repos/maria.zervou@databricks.com/Transformers4rec-databricks/transfromer4rec/dbfs:/merlin/data/processed/preproc_sessions_by_day/3",
        batch_size=1000,
        drop_last=False,
        shuffle=False,
        max_sequence_length=100
    )

# COMMAND ----------

batch=next(iter(train_loader))

# COMMAND ----------

batch[0]
