
from azureml.core import Workspace, Datastore, Dataset, Run
import pandas as pd
import os

# get the current run
run = Run.get_context()
ws = run.experiment.workspace

# get the dataset
ds = Dataset.get_by_name(ws, "diabetes")
diabetes_df = ds.to_pandas_dataframe()

# preprocessing here:
diabetes_df = diabetes_df[(diabetes_df["pres"] != 0) & (diabetes_df["mass"] != 0) & (diabetes_df["plas"] != 0)]

# save the data to csv
os.mkdir("data")
local_path = 'data/diabetes_cleaned.csv'
diabetes_df.to_csv(local_path, index=False)

# upload the data
datastore = ws.get_default_datastore()
datastore.upload(src_dir='data', target_path='cleaned', overwrite=True)

# create a dataset referencing the cloud location
dataset = Dataset.Tabular.from_delimited_files(path = [(datastore, ('cleaned/diabetes_cleaned.csv'))])

# register new dataset
diabetes_cleaned_ds = dataset.register(workspace=ws, name='diabetes_cleaned',description='Diabetes training data', create_new_version=True)
