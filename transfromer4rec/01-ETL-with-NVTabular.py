# Databricks notebook source
# Copyright 2022 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.

# COMMAND ----------

# MAGIC %md
# MAGIC # ETL with NVTabular
# MAGIC
# MAGIC **Start a GPU CLuster and run the below magic commmand**

# COMMAND ----------

# MAGIC %pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook demonstrates how to use NVTabular to perform the feature engineering that is needed to model the `YOOCHOOSE` dataset which contains a collection of sessions from a retailer. Each session  encapsulates the click events that the user performed in that session.
# MAGIC
# MAGIC First, let's start by importing several libraries:

# COMMAND ----------

import os
import glob
import numpy as np
import gc
import cudf
import cupy
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define Data Input and Output Paths

# COMMAND ----------

import os 

RAW_FOLDER = "/dbfs/merlin/data/raw" #where the raw data is stored
OUTPUT_PATH = "/local_disk0/merlin/data/" #where we will store the preprocessed data
OUTPUT_FOLDER  = os.path.join(OUTPUT_PATH,"output/")
OVERWRITE = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and clean raw data

# COMMAND ----------

interactions_df = cudf.read_parquet(RAW_FOLDER)

# COMMAND ----------

interactions_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remove repeated interactions within the same session

# COMMAND ----------

print("Count with in-session repeated interactions: {}".format(len(interactions_df)))

# Sorts the dataframe by session and timestamp, to remove consecutive repetitions
interactions_df.timestamp = interactions_df.timestamp.astype(int)
interactions_df = interactions_df.sort_values(['session_id', 'timestamp'])
past_ids = interactions_df['item_id'].shift(1).fillna()
session_past_ids = interactions_df['session_id'].shift(1).fillna()

# COMMAND ----------

# Keeping only in session interactions that are not repeated
interactions_df = interactions_df[~((interactions_df['session_id'] == session_past_ids) & (interactions_df['item_id'] == past_ids))]
print("Count after removed in-session repeated interactions: {}".format(len(interactions_df)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create new feature with the timestamp when the item was first seen

# COMMAND ----------

items_first_ts_df = interactions_df.groupby('item_id').agg({'timestamp': 'min'}).reset_index().rename(columns={'timestamp': 'itemid_ts_first'})
interactions_merged_df = interactions_df.merge(items_first_ts_df, on=['item_id'], how='left')
interactions_merged_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save the interactions_merged_df to disk to be able to use in the inference step.

# COMMAND ----------

# ! mkdir -p {OUTPUT_FOLDER}
!rm -rf  {OUTPUT_FOLDER}

# COMMAND ----------

interactions_merged_df.to_parquet(os.path.join(OUTPUT_FOLDER, 'interactions_merged_df/'),index = False,engine="pyarrow",row_group_size_bytes=67108864/4)

# COMMAND ----------

os.path.join(OUTPUT_FOLDER, 'interactions_merged_df.parquet')

# COMMAND ----------

# free gpu memory
del interactions_df, session_past_ids, items_first_ts_df,interactions_merged_df
gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Define a preprocessing workflow with NVTabular

# COMMAND ----------

# MAGIC %md
# MAGIC NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems. It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.
# MAGIC
# MAGIC NVTabular supports different feature engineering transformations required by deep learning (DL) models such as Categorical encoding and numerical feature normalization. It also supports feature engineering and generating sequential features. 
# MAGIC
# MAGIC More information about the supported features can be found <a href="https://nvidia-merlin.github.io/NVTabular/main/index.html> here. </a">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature engineering: Create and Transform items features

# COMMAND ----------

# MAGIC %md
# MAGIC In this cell, we are defining three transformations ops: 
# MAGIC
# MAGIC - 1. Encoding categorical variables using `Categorify()` op. We set `start_index` to 1 so that encoded null values start from `1` instead of `0` because we reserve `0` for padding the sequence features.
# MAGIC - 2. Deriving temporal features from timestamp and computing their cyclical representation using a custom lambda function. 
# MAGIC - 3. Computing the item recency in days using a custom op. Note that item recency is defined as the difference between the first occurrence of the item in dataset and the actual date of item interaction. 

# COMMAND ----------

# Encodes categorical features as contiguous integers
cat_feats = ColumnSelector(['session_id', 'category', 'item_id']) >> nvt.ops.Categorify(start_index=1)

# create time features
session_ts = ColumnSelector(['timestamp'])
session_time = (
    session_ts >> 
    nvt.ops.LambdaOp(lambda col: cudf.to_datetime(col, unit='s')) >> 
    nvt.ops.Rename(name = 'event_time_dt')
)
sessiontime_weekday = (
    session_time >> 
    nvt.ops.LambdaOp(lambda col: col.dt.weekday) >> 
    nvt.ops.Rename(name ='et_dayofweek')
)

# Derive cyclical features: Define a custom lambda function 
def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

weekday_sin = sessiontime_weekday >> (lambda col: get_cycled_feature_value_sin(col+1, 7)) >> nvt.ops.Rename(name = 'et_dayofweek_sin')

# Compute Item recency: Define a custom Op 
class ItemRecency(nvt.ops.Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            item_first_timestamp = gdf['itemid_ts_first']
            delta_days = (col - item_first_timestamp) / (60*60*24)
            gdf[column + "_age_days"] = delta_days * (delta_days >=0)
        return gdf

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        self._validate_matching_cols(input_schema, parents_selector, "computing input selector")
        return parents_selector

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[col_name + "_age_days"] = [col_name]
        return column_mapping

    @property
    def dependencies(self):
        return ["itemid_ts_first"]

    @property
    def output_dtype(self):
        return np.float64
    
recency_features = session_ts >> ItemRecency() 
# Apply standardization to this continuous feature
recency_features_norm = recency_features >> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(name='product_recency_days_log_norm')

time_features = (
    session_time +
    sessiontime_weekday +
    weekday_sin + 
    recency_features_norm
)

features = ColumnSelector(['timestamp', 'session_id']) + cat_feats + time_features 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the preprocessing of sequential features

# COMMAND ----------

# MAGIC %md
# MAGIC Once the item features are generated, the objective of this cell is to group interactions at the session level, sorting the interactions by time. We additionally truncate all sessions to first 20 interactions and filter out sessions with less than 2 interactions.

# COMMAND ----------

# Define Groupby Operator
groupby_features = features >> nvt.ops.Groupby(
    groupby_cols=["session_id"], 
    sort_cols=["timestamp"],
    aggs={
        'item_id': ["list", "count"],
        'category': ["list"],  
        'timestamp': ["first"],
        'event_time_dt': ["first"],
        'et_dayofweek_sin': ["list"],
        'product_recency_days_log_norm': ["list"]
        },
    name_sep="-") >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])


# Truncate sequence features to first interacted 20 items 
SESSIONS_MAX_LENGTH = 20 

groupby_features_list = groupby_features['item_id-list', 'category-list', 'et_dayofweek_sin-list', 'product_recency_days_log_norm-list']
groupby_features_truncated = groupby_features_list >> nvt.ops.ListSlice(0, SESSIONS_MAX_LENGTH, pad=True) >> nvt.ops.Rename(postfix = '_seq')

# Calculate session day index based on 'event_time_dt-first' column
day_index = ((groupby_features['event_time_dt-first'])  >> 
    nvt.ops.LambdaOp(lambda col: (col - col.min()).dt.days +1) >> 
    nvt.ops.Rename(f = lambda col: "day_index")
)

# Select features for training 
selected_features = groupby_features['session_id', 'item_id-count'] + groupby_features_truncated + day_index

# Filter out sessions with less than 2 interactions 
MINIMUM_SESSION_LENGTH = 2
filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["item_id-count"] >= MINIMUM_SESSION_LENGTH) 

# COMMAND ----------

# MAGIC %md
# MAGIC Avoid Numba low occupancy warnings:

# COMMAND ----------

from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute NVTabular workflow

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have defined the general workflow (`filtered_sessions`), we provide our cudf dataset to `nvt.Dataset` class which is optimized to split data into chunks that can fit in device memory and to handle the calculation of complex global statistics. Then, we execute the pipeline that fits and transforms data to get the desired output features.

# COMMAND ----------

dataset = nvt.Dataset(os.path.join(OUTPUT_FOLDER, 'interactions_merged_df/'),engine="parquet",part_size="16MB")
workflow = nvt.Workflow(filtered_sessions)
# Learn features statistics necessary of the preprocessing workflow
workflow.fit(dataset)
# Apply the preprocessing workflow in the dataset and convert the resulting Dask cudf dataframe to a cudf dataframe
sessions_gdf = workflow.transform(dataset).compute()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's print the head of our preprocessed dataset. You can notice that now each example (row) is a session and the sequential features with respect to user interactions were converted to lists with matching length.

# COMMAND ----------

sessions_gdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save the preprocessing workflow

# COMMAND ----------

workflow.save(os.path.join(OUTPUT_FOLDER,"workflow_etl"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export pre-processed data by day

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we are going to split the preprocessed parquet files by days, to allow for temporal training and evaluation. There will be a folder for each day and three parquet files within each day: `train.parquet`, `validation.parquet` and `test.parquet`.
# MAGIC   
# MAGIC P.s. It is worthwhile to note that the dataset has a single categorical feature (category), which, however, is inconsistent over time in the dataset. All interactions before day 84 (2014-06-23) have the same value for that feature, whereas many other categories are introduced afterwards. Thus for this example, we save only the last five days.

# COMMAND ----------

sessions_gdf = sessions_gdf[sessions_gdf.day_index>=59]

# COMMAND ----------

sessions_gdf.head()

# COMMAND ----------

from utils.date_utils import save_time_based_splits

# COMMAND ----------

# from utils.data_utils import save_time_based_splits
save_time_based_splits(data=nvt.Dataset(sessions_gdf),
                       output_dir= os.path.join(OUTPUT_FOLDER,"preproc_sessions_by_day"),
                       partition_col='day_index',
                       timestamp_col='session_id', 
                      )

# COMMAND ----------

def list_files(startpath):
    """
    Util function to print the nested structure of a directory
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))

# COMMAND ----------

list_files(os.path.join(OUTPUT_FOLDER,"preproc_sessions_by_day"))

# COMMAND ----------

# free gpu memory
del  sessions_gdf
gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC That's it! We created our sequential features, now we can go to the next notebook to train a PyTorch session-based model.
