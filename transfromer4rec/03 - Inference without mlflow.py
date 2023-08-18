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

import cloudpickle
import os 
import torch 

#Load the model class and its checkpoint
checkpoint_path = '/Workspace/Repos/maria.zervou@databricks.com/Transformers4rec-databricks/transfromer4rec/tmp/checkpoint-15490/'

loaded_model = cloudpickle.load(open(os.path.join(checkpoint_path, "t4rec_model_class.pkl"), "rb"))

#Restoring model weights
loaded_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))



# COMMAND ----------

from merlin_standard_lib import Schema
SCHEMA_PATH = "../schema_demo.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)

# COMMAND ----------

schema = schema.select_by_name(
   ['item_id-list_seq', 'category-list_seq', 'product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq']
)

# COMMAND ----------

import transformers4rec

  
#Set up data loader
def get_dataloader(data_path, batch_size):

  loader = transformers4rec.torch.utils.data_utils.MerlinDataLoader.from_schema(
      schema,
      data_path,
      batch_size,
      max_sequence_length=100,
      shuffle=False,
  )

  return loader

#Load 1000 obs from test set
loader=get_dataloader("/Workspace/Repos/maria.zervou@databricks.com/Transformers4rec-databricks/transfromer4rec/dbfs:/merlin/data/processed/preproc_sessions_by_day/3", 1000)

batch=next(iter(loader))



# COMMAND ----------

#Score model on this batch
response=loaded_model(batch[0], training=False)


# COMMAND ----------

import gc
gc.collect()

# COMMAND ----------

i=0
for data in loader:

  #  data = data.to('cuda')
  print(i)
  i=i+1
  del  data
  gc.collect()
  # response=loaded_model(data, training=False)



# COMMAND ----------

gc.collect()

# COMMAND ----------

df=spark.read.parquet("/Workspace/Repos/maria.zervou@databricks.com/Transformers4rec-databricks/transfromer4rec/dbfs:/merlin/data/processed/preproc_sessions_by_day/3/test.parquet")

# COMMAND ----------


