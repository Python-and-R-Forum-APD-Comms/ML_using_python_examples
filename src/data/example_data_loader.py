# Databricks notebook source
# from IPython.display import display
from DataLoader import DataLoader
from config import files

data_to_be_loaded = ["ecds"]

data = {}
for file in data_to_be_loaded:
    data[file] = DataLoader(files, file, spark)


# COMMAND ----------

display(data["ecds"].load_data())

# COMMAND ----------
