# Databricks notebook source
import train_model as tm
import pandas as pd
from sklearn.datasets import load_iris

# TODO :
# [X] functionalize this workbook
#

# COMMAND ----------

iris = load_iris()
data = pd.DataFrame(
    {
        "sepal length": iris.data[:, 0],
        "sepal width": iris.data[:, 1],
        "petal length": iris.data[:, 2],
        "petal width": iris.data[:, 3],
        "species": iris.target,
    }
)
data.head()

# COMMAND ----------

X = data[
    ["sepal length", "sepal width", "petal length", "petal width"]
]  # Features
Y = data["species"]  # Labels


# COMMAND ----------

clf_1, y_pred_1, y_pred_accuracy_1 = tm.basic_model_run(X, Y)
