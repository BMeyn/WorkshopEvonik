
from azureml.core import Run, Dataset

# get the workspace from the current run
run = Run.get_context()
ws = run.experiment.workspace
# get the default datastore
datastore = ws.get_default_datastore()

# # create a TabularDataset
datastore_paths = [(datastore,  'raw/diabetes.csv')]
diabetes_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
# diabetes_df = diabetes_ds.to_pandas_dataframe()

diabetes_ds = diabetes_ds.register(workspace=ws, name='diabetes',description='Diabetes training data', create_new_version=True)
