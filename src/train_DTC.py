

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.core.model import Model
import pandas as pd
import os
import joblib
import sklearn

# get the current run
run = Run.get_context()
ws = run.experiment.workspace

datastore = ws.get_default_datastore()

# get the dataset
ds = Dataset.get_by_name(ws, "diabetes_cleaned")
diabetes_df = ds.to_pandas_dataframe()


X = diabetes_df.drop("class", axis=1)
y = diabetes_df["class"]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_df["class"], random_state=0)

# init the model
model = DecisionTreeClassifier()

# train the model
model.fit(X_train, y_train)

# get the predictions
y_pred = model.predict(X_test)


# register a sample_input
sample_input = diabetes_df[:1].drop("class", axis=1)
sample_input = Dataset.Tabular.register_pandas_dataframe(name="diabetes_sample_input",target=datastore, dataframe=sample_input)

# register a sample_output
sample_output = pd.DataFrame({"class": y_pred[:1]})
sample_output = Dataset.Tabular.register_pandas_dataframe(name="diabetes_sample_output",target=datastore, dataframe=sample_output)

acc = accuracy_score(y_test, y_pred)
# log the accuracy to the Run
run.log("acc", acc)

# log confusion matrix
cmtx = sklearn.metrics.confusion_matrix(y_test,(y_pred))

cmtx_wrapper =  {
       "schema_type": "confusion_matrix",
       "schema_version": "v1",
       "data": {
           "class_labels": diabetes_df["class"].unique().tolist(),
           "matrix": cmtx.tolist()
       }
   }
   
run.log_confusion_matrix("confusion matrix", cmtx_wrapper, description='')

# save the model to disk
joblib.dump(model, 'model.pkl')

# register the model in the workspace
model_reg = Model.register(model_path="model.pkl",
                       model_name="sklearn-model",
                       model_framework=Model.Framework.SCIKITLEARN,
                       model_framework_version=sklearn.__version__,
                       sample_input_dataset=sample_input,
                       sample_output_dataset=sample_output,
                       tags={'area': "diabetes", 'type': "classification"},
                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       description="DecisionTreeClassifier for diabetes dataset",
                       workspace=ws)
